import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from unicore.data import (
    AppendTokenDataset,
    Dictionary,
    FromNumpyDataset,
    LMDBDataset,
    NestedDictionaryDataset,
    PrependTokenDataset,
    RawArrayDataset,
    RawLabelDataset,
    RightPadDataset,
    RightPadDataset2D,
    SortDataset,
    TokenizeDataset,
    data_utils,
)

from src.utils.torchtool import is_rank_zero

from .components.unimol import (
    AffinityDataset,
    CroppingPocketDockingPoseDataset,
    DistanceDataset,
    EdgeTypeDataset,
    KeyDataset,
    LengthDataset,
    NormalizeDataset,
    PrependAndAppend2DDataset,
    RemoveHydrogenDataset,
    RemoveHydrogenPocketDataset,
    ResamplingDataset,
    RightPadDatasetCoord,
)

logger = logging.getLogger(__name__)


class ProFSADataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_file: str = "valid.lmdb",
        mol_dict_file: str = "dict_mol.txt",
        pocket_dict_file: str = "dict_pocket.txt",
        max_pocket_atoms: int = 256,
        max_seq_len: int = 512,
        shuffle: bool = False,
        seed: int = 0,
        ligand_atoms_key="lig_atoms_real",
        ligand_coord_key="lig_coord_real",
        pocket_atoms_key="pocket_atoms",
        pocket_coord_key="pocket_coordinates",
        affinity_key="affinity",
    ):
        self.data_dir = Path(data_dir)
        self.data_path = str(self.data_dir / data_file)
        self.mol_dict_path = str(self.data_dir / mol_dict_file)
        self.pocket_dict_path = str(self.data_dir / pocket_dict_file)
        self.max_pocket_atoms = max_pocket_atoms
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.ligand_atoms_key = ligand_atoms_key
        self.ligand_coord_key = ligand_coord_key
        self.pocket_atoms_key = pocket_atoms_key
        self.pocket_coord_key = pocket_coord_key
        self.affinity_key = affinity_key

        self.mol_dict, self.pocket_dict = self.load_dictionary()
        self.dataset = self.load_dataset()

        if is_rank_zero():
            logger.info(
                f"{self.__class__.__name__}: {len(self)} samples in total."
            )

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        self.dataset.set_epoch(epoch)

    def load_dictionary(self):
        mol_dict = Dictionary.load(self.mol_dict_path)
        pocket_dict = Dictionary.load(self.pocket_dict_path)
        mol_dict.add_symbol("[MASK]", is_special=True)
        pocket_dict.add_symbol("[MASK]", is_special=True)
        if is_rank_zero():
            logger.info("mol dictionary: {} types".format(len(mol_dict)))
            logger.info("pocket dictionary: {} types".format(len(pocket_dict)))
        return mol_dict, pocket_dict

    def load_dataset(self):
        dataset = LMDBDataset(self.data_path)
        if self.shuffle:
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")
            dataset = AffinityDataset(
                dataset,
                self.seed,
                self.ligand_atoms_key,
                self.ligand_coord_key,
                self.pocket_atoms_key,
                self.pocket_coord_key,
                self.affinity_key,
                True,
            )
            tgt_dataset = KeyDataset(dataset, "affinity")

        else:
            dataset = AffinityDataset(
                dataset,
                self.seed,
                self.ligand_atoms_key,
                self.ligand_coord_key,
                self.pocket_atoms_key,
                self.pocket_coord_key,
                self.affinity_key,
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDockingPoseDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            "holo_pocket_coordinates",
            self.max_pocket_atoms,
        )

        dataset = RemoveHydrogenDataset(
            dataset, "atoms", "coordinates", True, True
        )

        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        mol_len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.mol_dict, max_seq_len=self.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.mol_dict.bos(), self.mol_dict.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.mol_dict))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        pocket_len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dict,
            max_seq_len=self.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dict.bos(),
            self.pocket_dict.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dict)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "mol_src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.mol_dict.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dict.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                    "mol_len": RawArrayDataset(mol_len_dataset),
                    "pocket_len": RawArrayDataset(pocket_len_dataset),
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
            },
        )
        if self.shuffle:
            with data_utils.numpy_seed(self.seed):
                shuffle = np.random.permutation(len(src_dataset))

            dataset = SortDataset(
                dataset,
                sort_order=[shuffle],
            )
            dataset = ResamplingDataset(dataset)
        return dataset


class ProFSADataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        pin_memory: bool = False,
        batch_size: Union[int, Dict[str, int]] = None,
        dataset_cfg: Dict[str, Any] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

    def _dataloader(self, split):
        dataset_cfg = self.hparams["dataset_cfg"][split]
        dataset = ProFSADataset(**dataset_cfg)
        if type(self.hparams["batch_size"]) == int:
            batch_size = self.hparams.batch_size
        else:
            batch_size = self.hparams.batch_size[split]
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.dataset.collater,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        if "test" not in self.hparams.dataset_cfg:
            super().test_dataloader()
        return self._dataloader("test")
