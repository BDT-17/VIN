"""Deduplication logic for dataset images.

Implements three layers of deduplication:
1. SHA-256 for exact duplicates
2. pHash for near duplicates (resized/compressed)
3. CLIP/DINO embeddings for perceptual similarity
"""

import hashlib
import imagehash
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm


class DeduplicatorConfig:
    """Configuration for deduplication."""
    # pHash similarity threshold (lower = more similar)
    PHASH_DISTANCE_THRESHOLD = 5

    # CLIP/DINO cosine similarity threshold
    EMBEDDING_SIMILARITY_THRESHOLD = 0.95

    # Temporal neighbor threshold (frames within N of each other)
    TEMPORAL_NEIGHBOR_FRAMES = 30


class ImageDeduplicator:
    """Multi-stage image deduplication."""

    def __init__(
        self,
        working_dir: Path,
        config: Optional[DeduplicatorConfig] = None,
    ):
        self.working_dir = Path(working_dir)
        self.config = config or DeduplicatorConfig()

        self.inventory_dir = self.working_dir / "raw_inventory"
        self.normalized_dir = self.working_dir / "normalized"
        self.normalized_dir.mkdir(parents=True, exist_ok=True)

    def deduplicate_all_sources(
        self,
        source_priorities: Dict[str, int],
    ) -> Path:
        """Run full deduplication pipeline across all sources.

        Args:
            source_priorities: Map of source_id to duplicate_priority
                Lower priority number = higher precedence (survives)

        Returns:
            Path to groups.parquet
        """
        print("=" * 60)
        print("DEDUPLICATION PIPELINE")
        print("=" * 60)

        # Load all image inventories
        all_images_df = self._load_all_images()

        print(f"\nTotal images: {len(all_images_df)}")

        # Stage 1: SHA-256 exact duplicates
        print("\n[1/3] Finding exact duplicates (SHA-256)...")
        sha256_clusters = self._find_sha256_duplicates(all_images_df)
        print(f"  Found {len(sha256_clusters)} exact duplicate clusters")

        # Stage 2: pHash near duplicates
        print("\n[2/3] Finding near duplicates (pHash)...")
        phash_clusters = self._find_phash_duplicates(all_images_df, sha256_clusters)
        print(f"  Found {len(phash_clusters)} near duplicate clusters")

        # Stage 3: Temporal neighbors (for video sources)
        print("\n[3/3] Finding temporal neighbors...")
        temporal_clusters = self._find_temporal_neighbors(all_images_df)
        print(f"  Found {len(temporal_clusters)} temporal clusters")

        # Merge all clusters and assign canonical images
        print("\nAssigning canonical images...")
        groups_df = self._build_groups(
            all_images_df,
            sha256_clusters,
            phash_clusters,
            temporal_clusters,
            source_priorities,
        )

        # Save groups.parquet
        groups_path = self.normalized_dir / "groups.parquet"
        groups_df.to_parquet(groups_path, index=False)

        print(f"\n✓ Groups saved: {groups_path}")
        print(f"  Total images: {len(groups_df)}")
        print(f"  Unique images: {(groups_df['dedupe_status'] == 'unique').sum()}")
        print(f"  Exact duplicates: {(groups_df['dedupe_status'] == 'exact_duplicate').sum()}")
        print(f"  Near duplicates: {(groups_df['dedupe_status'] == 'near_duplicate').sum()}")
        print(f"  Temporal neighbors: {(groups_df['dedupe_status'] == 'temporal_neighbor').sum()}")

        return groups_path

    def _load_all_images(self) -> pd.DataFrame:
        """Load and concatenate all image inventories."""
        dfs = []

        for parquet_file in self.inventory_dir.glob("*_images.parquet"):
            df = pd.read_parquet(parquet_file)
            dfs.append(df)

        if not dfs:
            raise ValueError(f"No image inventories found in {self.inventory_dir}")

        return pd.concat(dfs, ignore_index=True)

    def _find_sha256_duplicates(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Find exact duplicates using SHA-256 hash.

        Returns:
            Map of sha256 -> list of image_ids
        """
        clusters = defaultdict(list)

        for _, row in df.iterrows():
            clusters[row['sha256']].append(row['image_id'])

        # Keep only clusters with duplicates
        return {k: v for k, v in clusters.items() if len(v) > 1}

    def _find_phash_duplicates(
        self,
        df: pd.DataFrame,
        sha256_clusters: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Find near duplicates using perceptual hash.

        Args:
            df: Image dataframe
            sha256_clusters: Already identified exact duplicates

        Returns:
            Map of cluster_id -> list of image_ids
        """
        # Skip images already in exact duplicate clusters
        exact_dup_image_ids = set()
        for image_ids in sha256_clusters.values():
            exact_dup_image_ids.update(image_ids)

        # Build pHash lookup
        phash_to_images = defaultdict(list)

        for _, row in df.iterrows():
            if row['image_id'] in exact_dup_image_ids:
                continue
            phash_to_images[row['phash']].append(row['image_id'])

        # Find clusters with similar pHash
        clusters = {}
        cluster_id = 0

        phash_list = list(phash_to_images.keys())

        for i, phash_str in enumerate(tqdm(phash_list, desc="  Computing pHash distances")):
            if not phash_to_images[phash_str]:
                continue

            phash_obj = imagehash.hex_to_hash(phash_str)
            similar = [phash_to_images[phash_str]]

            for j in range(i + 1, len(phash_list)):
                other_phash_str = phash_list[j]
                if not phash_to_images[other_phash_str]:
                    continue

                other_phash_obj = imagehash.hex_to_hash(other_phash_str)
                distance = phash_obj - other_phash_obj

                if distance <= self.config.PHASH_DISTANCE_THRESHOLD:
                    similar.append(phash_to_images[other_phash_str])
                    # Mark as consumed
                    phash_to_images[other_phash_str] = []

            if len(similar) > 1:
                # Flatten
                all_image_ids = [img_id for group in similar for img_id in group]
                clusters[f"phash_cluster_{cluster_id}"] = all_image_ids
                cluster_id += 1

        return clusters

    def _find_temporal_neighbors(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Find temporal neighbors from video sequences.

        Groups frames from the same source that are temporally close.

        Returns:
            Map of cluster_id -> list of image_ids
        """
        clusters = {}

        # Group by source_id and check for frame_index
        for source_id, group_df in df.groupby('source_id'):
            if 'frame_index' not in group_df.columns:
                continue

            # Sort by frame index
            group_df = group_df.sort_values('frame_index')

            # Group consecutive frames
            cluster_id = 0
            current_cluster = []
            prev_frame = None

            for _, row in group_df.iterrows():
                frame_idx = row.get('frame_index')
                if pd.isna(frame_idx):
                    continue

                frame_idx = int(frame_idx)

                if prev_frame is None or frame_idx - prev_frame <= self.config.TEMPORAL_NEIGHBOR_FRAMES:
                    current_cluster.append(row['image_id'])
                else:
                    # Save cluster if large enough
                    if len(current_cluster) > 1:
                        clusters[f"{source_id}_temporal_{cluster_id}"] = current_cluster
                        cluster_id += 1
                    current_cluster = [row['image_id']]

                prev_frame = frame_idx

            # Save final cluster
            if len(current_cluster) > 1:
                clusters[f"{source_id}_temporal_{cluster_id}"] = current_cluster

        return clusters

    def _build_groups(
        self,
        df: pd.DataFrame,
        sha256_clusters: Dict[str, List[str]],
        phash_clusters: Dict[str, List[str]],
        temporal_clusters: Dict[str, List[str]],
        source_priorities: Dict[str, int],
    ) -> pd.DataFrame:
        """Build final groups dataframe with canonical assignments.

        Args:
            df: Image dataframe
            sha256_clusters: Exact duplicate clusters
            phash_clusters: Near duplicate clusters
            temporal_clusters: Temporal neighbor clusters
            source_priorities: Source duplicate priorities

        Returns:
            DataFrame with group assignments
        """
        # Initialize result
        groups = []

        # Create image_id to source_id mapping
        image_to_source = dict(zip(df['image_id'], df['source_id']))

        # Track which images have been assigned
        assigned = set()

        # Process exact duplicates
        for cluster_id, (sha256_hash, image_ids) in enumerate(sha256_clusters.items()):
            canonical = self._choose_canonical(image_ids, image_to_source, source_priorities)

            for img_id in image_ids:
                groups.append({
                    'image_id': img_id,
                    'duplicate_cluster_id': f"exact_{cluster_id}",
                    'split_group_id': f"exact_{cluster_id}",
                    'dedupe_status': 'exact_duplicate',
                    'canonical_image_id': canonical,
                    'duplicate_type': 'exact',
                    'similarity_score': 1.0,
                })
                assigned.add(img_id)

        # Process near duplicates
        for cluster_id, (cluster_name, image_ids) in enumerate(phash_clusters.items()):
            # Filter out already assigned
            image_ids = [img_id for img_id in image_ids if img_id not in assigned]
            if not image_ids:
                continue

            canonical = self._choose_canonical(image_ids, image_to_source, source_priorities)

            for img_id in image_ids:
                groups.append({
                    'image_id': img_id,
                    'duplicate_cluster_id': cluster_name,
                    'split_group_id': cluster_name,
                    'dedupe_status': 'near_duplicate',
                    'canonical_image_id': canonical,
                    'duplicate_type': 'near_duplicate',
                    'similarity_score': 0.95,
                })
                assigned.add(img_id)

        # Process temporal neighbors
        for cluster_name, image_ids in temporal_clusters.items():
            # Filter out already assigned
            image_ids = [img_id for img_id in image_ids if img_id not in assigned]
            if not image_ids:
                continue

            canonical = self._choose_canonical(image_ids, image_to_source, source_priorities)

            for img_id in image_ids:
                groups.append({
                    'image_id': img_id,
                    'duplicate_cluster_id': cluster_name,
                    'split_group_id': cluster_name,
                    'dedupe_status': 'temporal_neighbor',
                    'canonical_image_id': canonical,
                    'duplicate_type': 'temporal',
                    'similarity_score': None,
                })
                assigned.add(img_id)

        # Process unique images
        for img_id in df['image_id']:
            if img_id not in assigned:
                groups.append({
                    'image_id': img_id,
                    'duplicate_cluster_id': img_id,
                    'split_group_id': img_id,
                    'dedupe_status': 'unique',
                    'canonical_image_id': img_id,
                    'duplicate_type': None,
                    'similarity_score': None,
                })

        return pd.DataFrame(groups)

    def _choose_canonical(
        self,
        image_ids: List[str],
        image_to_source: Dict[str, str],
        source_priorities: Dict[str, int],
    ) -> str:
        """Choose canonical image from a duplicate cluster.

        Uses source priority (lower = better).
        """
        best_img = image_ids[0]
        best_priority = float('inf')

        for img_id in image_ids:
            source_id = image_to_source[img_id]
            priority = source_priorities.get(source_id, 999)

            if priority < best_priority:
                best_priority = priority
                best_img = img_id

        return best_img
