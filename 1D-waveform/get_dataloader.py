import datetime
import torch
from ASVRawDataset import ASVRawDataset
from pathlib import Path
from DummyDataset import DummyDataset
from customize_collate_fn import customize_collate
from customize_sampler import SamplerBlockShuffleByLen
from torch.utils.data.dataloader import DataLoader

"""
A simple and naive replica of the dataloader described in [1] to 
handle variable length input for ASVspoof 2019 LA database.

[1] X. Wang and J. Yamagishi, “A comparative study on recent neural spoofing countermeasures for synthetic speech
detection,” in Proc. Interspeech, 2021, pp. 4259–4263.
Codes of [1]: https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/core_scripts/data_io
Useful link: https://pytorch.org/docs/stable/data.html
"""


def get_dataloader(args, protocol_names):
    root_path = Path(args.data)
    if args.is_fixed_length:
        # to load each audio file with fixed length as usual
        print("Reading datasets with fixed length...")
    
        train_dataset = ASVRawDataset(Path(args.data), 'train', protocol_names['train_protocol'], is_rand=args.is_rand, handler=args.handler)
        dev_dataset = ASVRawDataset(Path(args.data), 'dev', protocol_names['dev_protocol'], is_rand=args.is_rand, handler=args.handler)
        eval_dataset = ASVRawDataset(Path(args.data), 'eval', protocol_names['eval_protocol'], is_rand=args.is_rand, handler=args.handler)
        train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        )
        dev_loader = torch.utils.data.DataLoader(
            dataset=dev_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        eval_loader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        return train_loader, dev_loader, eval_loader

    else:
        # to load variable length input
        print("Reading datasets with variable length...")

        # these datasets are just to get the audio length of each file provided in the given protocols
        train_dummy_dataset = DummyDataset(Path(args.data), 'train', protocol_names['train_protocol'])
        dev_dummy_dataset = DummyDataset(Path(args.data), 'dev', protocol_names['dev_protocol'])
        eval_dummy_dataset = DummyDataset(Path(args.data), 'eval', protocol_names['eval_protocol'])
    
        train_dummy_loader = torch.utils.data.DataLoader(
            dataset=train_dummy_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            # have to set shuffle=False to generate the length list in correct (orginal) order
            shuffle=False,
        )
        dev_dummy_loader = torch.utils.data.DataLoader(
            dataset=dev_dummy_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            shuffle=False,
        )
        eval_dummy_loader = torch.utils.data.DataLoader(
            dataset=eval_dummy_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            shuffle=False,
        )

        print("Listing train data base...")
        a = datetime.datetime.now()
        # get the list of audio length
        train_seq_len_list = [audio_length for file_name, audio_length in train_dummy_loader]
        # get the customised sampler to sample audio files with similar length to form a mini-batch
        train_dummy_sampler = SamplerBlockShuffleByLen(train_seq_len_list, args.batch_size)

        dummy_collate_fn = customize_collate

        b = datetime.datetime.now()
        print("Done")
        print("Time cost: ", b-a)

        print("Listing dev data base...")
        a = datetime.datetime.now()
        dev_seq_len_list = [audio_length for file_name, audio_length in dev_dummy_loader]
        dev_dummy_sampler = SamplerBlockShuffleByLen(dev_seq_len_list, args.batch_size)
        b = datetime.datetime.now()
        print("Done")
        print("Time cost: ", b-a)

        print("Listing eval data base...")
        a = datetime.datetime.now()
        eval_seq_len_list = [audio_length for file_name, audio_length in eval_dummy_loader]
        eval_dummy_sampler = SamplerBlockShuffleByLen(eval_seq_len_list, args.batch_size)
        b = datetime.datetime.now()
        print("Done")
        print("Time cost: ", b-a)

        train_dataset = ASVRawDataset(Path(args.data), 'train', protocol_names['train_protocol'], is_rand=args.is_rand, is_fixed_length=False, handler=args.handler)
        dev_dataset = ASVRawDataset(Path(args.data), 'dev', protocol_names['dev_protocol'], is_rand=args.is_rand, is_fixed_length=False, handler=args.handler)
        eval_dataset = ASVRawDataset(Path(args.data), 'eval', protocol_names['eval_protocol'], is_rand=args.is_rand, is_fixed_length=False, handler=args.handler)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            # use customised sampler
            sampler=train_dummy_sampler,
            num_workers=args.num_workers,
            # a customised collate_fn is used to pad the audio files to 
            # have the same length (number of points),
            # so that we can use batch_size > 1 for variable length inputs
            collate_fn=dummy_collate_fn,
        )
        dev_loader = torch.utils.data.DataLoader(
            dataset=dev_dataset,
            batch_size=args.batch_size,
            sampler=dev_dummy_sampler,
            num_workers=args.num_workers,
            collate_fn=dummy_collate_fn,
        )
        eval_loader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=args.batch_size,
            sampler=eval_dummy_sampler,
            num_workers=args.num_workers,
            collate_fn=dummy_collate_fn,
        )
        return train_loader, dev_loader, eval_loader