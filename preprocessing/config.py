from pathlib import Path
from pydantic import BaseModel, Field, field_validator, computed_field
import json


class Config(BaseModel):
    """Global configuration for preprocessing pipeline."""

    # base / project
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Project root directory",
    )

    # length filtering
    min_rna_length: int = Field(default=50, ge=1, le=10000)
    max_rna_length: int = Field(default=500, ge=1, le=10000)

    # CD-HIT params
    cd_hit_identity_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    cdhit_word_size: int = Field(default=8, ge=4, le=10)
    cdhit_memory: int = Field(default=0, ge=0)  # 0 = unlimited
    cdhit_threads: int = Field(default=0, ge=0)  # 0 = all CPUs

    # data quality / weights
    experimental_quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    predicted_quality_score: float = Field(default=0.8, ge=0.0, le=1.0)

    # pipeline toggles
    build_graphs: bool = Field(
        default=True,
        description="Generate RNA graph representations during preprocessing.",
    )
    build_motif_membership: bool = Field(
        default=True,
        description="Compute motif membership features during preprocessing.",
    )
    allow_pseudoknots: bool = Field(
        default=True,
        description="Preserve pseudoknot information when parsing structures.",
    )

    # splits
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
    random_seed: int = Field(default=42, ge=0)

    # eval
    n_splits: int = Field(default=5, ge=1, le=20)

    @field_validator("max_rna_length")
    @classmethod
    def validate_max_greater_than_min(cls, v, info):
        """Ensure max_rna_length > min_rna_length."""
        min_len = info.data.get("min_rna_length")
        if min_len is not None and v <= min_len:
            raise ValueError(
                f"max_rna_length ({v}) must be > min_rna_length ({min_len})"
            )
        return v

    def model_post_init(self, __context) -> None:
        """Validate that split ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # computed paths

    @computed_field
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @computed_field
    @property
    def data_raw_dir(self) -> Path:
        return self.data_dir / "unzipped"

    @computed_field
    @property
    def data_intermediate_dir(self) -> Path:
        return self.data_dir / "intermediate"

    @computed_field
    @property
    def data_processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @computed_field
    @property
    def data_graphs_dir(self) -> Path:
        return self.data_dir / "graphs"

    @computed_field
    @property
    def fasta_file(self) -> Path:
        return self.data_raw_dir / "bpRNA_1m_90.fasta"

    @computed_field
    @property
    def bpseq_dir(self) -> Path:
        return self.data_raw_dir / "bpRNA_1m_90_BPSEQFILES"

    @computed_field
    @property
    def dbn_dir(self) -> Path:
        return self.data_raw_dir / "bpRNA_1m_90_DBNFILES"

    @computed_field
    @property
    def st_dir(self) -> Path:
        return self.data_raw_dir / "bpRNA_1m_90_STFILES"

    # helper methods

    def create_directories(self):
        """Create necessary derived data directories if they don't exist."""
        self.data_intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.data_graphs_dir.mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> bool:
        """Validate that required raw data paths exist."""
        required_paths = [
            self.fasta_file,
            self.bpseq_dir,
            self.dbn_dir,
            self.st_dir,
        ]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required path not found: {path}")
        return True

    def save_as_json(self, path: Path | str):
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

    @classmethod
    def load_from_json(cls, path: Path | str) -> "Config":
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    model_config = {
        "frozen": False,
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


def get_default_config() -> Config:
    """Return a new default Config instance safely."""
    return Config()


# HACK: shared default instance for convenience imports
DEFAULT_CONFIG: Config = get_default_config()
