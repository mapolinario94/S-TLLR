import numpy as np
import torch
from torchvision import transforms, datasets
import tonic
import logging
# from torchtoolbox.transform import Cutout
from .cifar10_dvs import CIFAR10DVS
from .augmentation import ToPILImage, Resize, Padding, RandomCrop, ToTensor, Normalize, RandomHorizontalFlip


def dataloader(args, dataset='DVSGesture', evaluate=False, distributed=False, batch_size=16, val_batch_size=16, workers=4):
    data_path = args.data_path
    if dataset == 'DVSGesture':
        train_loader, val_loader, trainset_len, testset_len = dataloader_gesture(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 11
        args.n_steps = 20
        args.n_inputs = 2
        args.dt = 75e-3
        args.classif = True
        args.delay_targets = 5
        args.skip_test = False
    elif dataset == 'DVSGesture100':
        train_loader, val_loader, trainset_len, testset_len = dataloader_gesture100(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 11
        args.n_steps = 20
        args.n_inputs = 2
        args.dt = 7.5e-3
        args.classif = True
        args.delay_targets = 5
        args.skip_test = False
    elif dataset == 'SHD':
        train_loader, val_loader, trainset_len, testset_len = dataloader_shd(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 20
        args.n_steps = 100
        args.n_inputs = 700
        args.dt = 4e-3
        args.classif = True
        args.delay_targets = 50
        args.skip_test = False
    elif dataset == 'MNIST':
        train_loader, val_loader, trainset_len, testset_len = dataloader_mnist(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 10
        args.n_steps = 20
        args.n_inputs = 28
        args.dt = 1e-3
        args.classif = True
        args.delay_targets = 5  # 5
        args.skip_test = False
    elif dataset == 'RowMNIST':
        train_loader, val_loader, trainset_len, testset_len = dataloader_mnist(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 10
        args.n_steps = 28
        args.n_inputs = 28
        args.dt = 1e-3
        args.classif = True
        args.delay_targets = 5  # 5
        args.skip_test = False
    elif dataset == "NMNIST":  # Dim: (2, 34, 34)
        train_loader, val_loader, trainset_len, testset_len = dataloader_nmnist(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 10
        args.n_steps = 30
        args.n_inputs = 2
        args.dt = 10e-3
        args.classif = True
        args.delay_targets = 5
        args.skip_test = False
    elif dataset == "NCALTECH101":  # Dim: (2, 34, 34)
        train_loader, val_loader, trainset_len, testset_len = dataloader_NCALTECH101(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 101
        args.n_steps = 10
        args.n_inputs = 2
        args.dt = 30e-3
        args.classif = True
        args.delay_targets = 10
        args.skip_test = False
    elif dataset == "CIFAR10DVS":  # Dim: (2, 34, 34)
        train_loader, val_loader, trainset_len, testset_len = dataloader_cifar10dvs(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 10
        args.n_steps = 10
        args.n_inputs = 2
        args.dt = 10e-3
        args.classif = True
        args.delay_targets = 7
        args.skip_test = False
    else:
        logging.info("ERROR: {0} is not supported".format(dataset))
        raise NameError("{0} is not supported".format(dataset))

    return train_loader, val_loader


def dataloader_shd(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets"):
    sensor_size = tonic.datasets.SHD.sensor_size
    transform = tonic.transforms.ToFrame(
        sensor_size=sensor_size,
        time_window=10000,
    )
    trainset = tonic.datasets.SHD(save_to=data_path, train=True)
    testset = tonic.datasets.SHD(save_to=data_path, train=False)

    slicing_time_window = 1000000  # microseconds
    slicer = tonic.slicers.SliceByTime(time_window=slicing_time_window)
    sliced_trainset = tonic.SlicedDataset(
        trainset, slicer=slicer, metadata_path=data_path + '/SHD/online_sliced_train', transform=transform
    )
    sliced_testset = tonic.SlicedDataset(
        testset, slicer=slicer, metadata_path=data_path + '/SHD/online_sliced_test', transform=transform
    )
    print(
        f"Went from {len(trainset)} samples in the original dataset to {len(sliced_trainset)} in the sliced version.")
    print(
        f"Went from {len(testset)} samples in the original dataset to {len(sliced_testset)} in the sliced version.")

    train_loader = torch.utils.data.DataLoader(sliced_trainset, batch_size=batch_size, shuffle=True,
                                               collate_fn=tonic.collation.PadTensors(batch_first=True), num_workers=workers)
    test_loader = torch.utils.data.DataLoader(sliced_testset, batch_size=val_batch_size, shuffle=False,
                                               collate_fn=tonic.collation.PadTensors(batch_first=True), num_workers=workers)

    return train_loader, test_loader, len(trainset), len(testset)


def dataloader_gesture(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets", reproducibility=False):
    labels = 11
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    trainset_ori = tonic.datasets.DVSGesture(save_to=data_path, train=True)
    testset_ori = tonic.datasets.DVSGesture(save_to=data_path, train=False)

    slicing_time_window = 1575000
    slicer = tonic.slicers.SliceByTime(time_window=slicing_time_window)

    frame_transform = tonic.transforms.Compose([  # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=75000),
        torch.tensor, transforms.Resize(32)
    ])
    frame_transform_test = tonic.transforms.Compose([  # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size,
                                 time_window=75000),
        torch.tensor,
        transforms.Resize(32, antialias=True)
    ])

    trainset_ori_sl = tonic.SlicedDataset(trainset_ori, slicer=slicer,
                                          metadata_path=data_path + '/metadata/online_dvsg_train',
                                          transform=frame_transform)
    # testset_ori_sl = tonic.SlicedDataset(testset_ori, slicer=slicer,
    #                                      metadata_path=data_path + '/metadata/online_dvsg_test',
    #                                      transform=frame_transform_test)

    print(
        f"Went from {len(trainset_ori)} samples in the original dataset to {len(trainset_ori_sl)} in the sliced version.")
    print(
        f"Went from {len(testset_ori)} samples in the original dataset to {len(testset_ori)} in the sliced version.")

    frame_transform2 = tonic.transforms.Compose([  # tonic.transforms.DropEvent(p=0.1),
        torch.tensor,
        transforms.RandomCrop(32, padding=4)
    ])

    trainset = tonic.CachedDataset(trainset_ori_sl,
                                   cache_path=data_path + '/cache/online_fast_dataloading_train',
                                   transform=frame_transform2)
    # if evaluate:
    testset = tonic.CachedDataset(testset_ori,
                                  cache_path=data_path + '/cache/online_fast_dataloading_test',
                                  transform=frame_transform_test)

    if reproducibility:
        import numpy as np
        import random
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,
            collate_fn=tonic.collation.PadTensors(batch_first=True), worker_init_fn=seed_worker, generator=g, )
        val_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=val_batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            collate_fn=tonic.collation.PadTensors(batch_first=True), worker_init_fn=seed_worker, generator=g, )
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,
            collate_fn=tonic.collation.PadTensors(batch_first=True))
        val_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=val_batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            collate_fn=tonic.collation.PadTensors(batch_first=True))

    return train_loader, val_loader, len(trainset_ori_sl), len(testset_ori)


def str_to_num(x):
    labels_dict = {'cup': 0, 'ibis': 1, 'crocodile': 2, 'wild_cat': 3, 'Leopards': 4, 'watch': 5, 'pagoda': 6, 'soccer_ball': 7, 'accordion': 8, 'sunflower': 9, 'rooster': 10, 'ewer': 11, 'stegosaurus': 12, 'ketch': 13, 'rhino': 14, 'cellphone': 15, 'brontosaurus': 16, 'buddha': 17, 'chandelier': 18, 'crayfish': 19, 'strawberry': 20, 'stapler': 21, 'nautilus': 22, 'stop_sign': 23, 'BACKGROUND_Google': 24, 'lamp': 25, 'platypus': 26, 'gerenuk': 27, 'starfish': 28, 'octopus': 29, 'flamingo_head': 30, 'butterfly': 31, 'revolver': 32, 'umbrella': 33, 'garfield': 34, 'sea_horse': 35, 'yin_yang': 36, 'beaver': 37, 'metronome': 38, 'tick': 39, 'trilobite': 40, 'airplanes': 41, 'hawksbill': 42, 'chair': 43, 'pizza': 44, 'anchor': 45, 'euphonium': 46, 'lotus': 47, 'minaret': 48, 'cannon': 49, 'bonsai': 50, 'windsor_chair': 51, 'wrench': 52, 'headphone': 53, 'Motorbikes': 54, 'scorpion': 55, 'cougar_face': 56, 'crocodile_head': 57, 'mandolin': 58, 'barrel': 59, 'inline_skate': 60, 'ferry': 61, 'laptop': 62, 'bass': 63, 'okapi': 64, 'saxophone': 65, 'hedgehog': 66, 'cougar_body': 67, 'scissors': 68, 'crab': 69, 'dalmatian': 70, 'dolphin': 71, 'mayfly': 72, 'pigeon': 73, 'emu': 74, 'electric_guitar': 75, 'panda': 76, 'helicopter': 77, 'schooner': 78, 'camera': 79, 'ant': 80, 'water_lilly': 81, 'elephant': 82, 'llama': 83, 'car_side': 84, 'binocular': 85, 'ceiling_fan': 86, 'menorah': 87, 'dragonfly': 88, 'brain': 89, 'joshua_tree': 90, 'lobster': 91, 'grand_piano': 92, 'flamingo': 93, 'wheelchair': 94, 'dollar_bill': 95, 'kangaroo': 96, 'gramophone': 97, 'Faces_easy': 98, 'snoopy': 99, 'pyramid': 100}
    return torch.tensor(labels_dict[x])


def dataloader_NCALTECH101(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets"):

    sensor_size = (240, 180, 2)
    frame_transform = tonic.transforms.Compose([  # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=30000),
        torch.tensor, transforms.Resize([45, 60])
    ])
    trainset_ori = tonic.datasets.NCALTECH101(save_to=data_path, transform=frame_transform,
                                              target_transform=str_to_num)

    dataset_len = len(trainset_ori)
    train_len = int(dataset_len*0.9)
    trainset, testset = torch.utils.data.random_split(trainset_ori, [train_len, dataset_len - train_len],
                                               generator=torch.Generator().manual_seed(42))

    trainset = tonic.CachedDataset(trainset,
                                   cache_path=data_path + '/cache/online_fast_dataloading_train_ncal')
    # if evaluate:
    testset = tonic.CachedDataset(testset,
                                  cache_path=data_path + '/cache/online_fast_dataloading_test_ncal')


    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
        collate_fn=tonic.collation.PadTensors(batch_first=True))

    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=val_batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=tonic.collation.PadTensors(batch_first=True))

    return train_loader, val_loader, len(trainset), len(testset)


def dataloader_gesture100(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets"):
    labels = 11
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    trainset_ori = tonic.datasets.DVSGesture(save_to=data_path, train=True)
    testset_ori = tonic.datasets.DVSGesture(save_to=data_path, train=False)
    test_len = len(testset_ori)

    slicing_time_window = 1575000
    slicer = tonic.slicers.SliceByTime(time_window=slicing_time_window)

    frame_transform = tonic.transforms.Compose([  # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=75000),
        torch.tensor, transforms.Resize(64)
    ])
    frame_transform_test = tonic.transforms.Compose([  # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size,
                                 time_window=75000),
        torch.tensor,
        transforms.Resize(64)
    ])

    trainset_ori_sl = tonic.SlicedDataset(trainset_ori, slicer=slicer,
                                          metadata_path=data_path + '/metadata/online_dvsg_train100',
                                          transform=frame_transform)

    testset_ori_sl = tonic.SlicedDataset(testset_ori, slicer=slicer,
                                         metadata_path=data_path + '/metadata/online_dvsg_test100',
                                         transform=frame_transform_test)

    print(
        f"Went from {len(trainset_ori)} samples in the original dataset to {len(trainset_ori_sl)} in the sliced version.")
    print(
        f"Went from {len(testset_ori)} samples in the original dataset to {len(testset_ori_sl)} in the sliced version.")

    frame_transform2 = tonic.transforms.Compose([  # tonic.transforms.DropEvent(p=0.1),
        torch.tensor,
        transforms.RandomCrop(64, padding=4)
    ])

    trainset = tonic.CachedDataset(trainset_ori_sl,
                                   cache_path=data_path + '/cache/online_fast_dataloading_train100',
                                   transform=frame_transform2)
    testset = tonic.CachedDataset(testset_ori_sl,
                                  cache_path=data_path + '/cache/online_fast_dataloading_test100')

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True,
        collate_fn=tonic.collation.PadTensors(batch_first=True) )

    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=val_batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
        collate_fn=tonic.collation.PadTensors(batch_first=True) )

    return train_loader, val_loader, len(trainset_ori_sl), len(testset_ori_sl)


def dataloader_mnist(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets"):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )

    testset = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=workers)

    return train_loader, val_loader, len(trainset), len(testset)


def dataloader_nmnist(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets"):
    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = tonic.transforms.Compose([
        torch.tensor,
        transforms.RandomCrop(34, padding=4)
    ])
    frame_transform_test = tonic.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=10000),
    ])

    trainset_ori = tonic.datasets.NMNIST(save_to=data_path, transform=frame_transform_test, train=True)
    testset = tonic.datasets.NMNIST(save_to=data_path, transform=frame_transform_test, train=False)

    trainset = tonic.DiskCachedDataset(
        trainset_ori,
        cache_path=data_path + '/NMNIST/cache/online_fast_dataloading_train',
        transform=frame_transform
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, collate_fn=tonic.collation.PadTensors(batch_first=True),
        num_workers=workers
    )
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=val_batch_size, shuffle=False, collate_fn=tonic.collation.PadTensors(batch_first=True),
        num_workers=workers
    )

    return train_loader, val_loader, len(trainset), len(testset)


def dataloader_cifar10dvs(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets", img_size=48):
    transform_train = transforms.Compose([
        ToPILImage(),
        Resize(48),
        Padding(4),
        RandomCrop(size=48, consistent=True),
        ToTensor(),
        Normalize((0.2728, 0.1295), (0.2225, 0.1290)),
    ])

    transform_test = transforms.Compose([
        ToPILImage(),
        Resize(48),
        ToTensor(),
        Normalize((0.2728, 0.1295), (0.2225, 0.1290)),
    ])
    num_classes = 10

    trainset = CIFAR10DVS(data_path, train=True, use_frame=True, frames_num=10, split_by='number',
                          normalization=None, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)

    testset = CIFAR10DVS(data_path, train=False, use_frame=True, frames_num=10, split_by='number',
                         normalization=None, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=workers)

    return train_loader, val_loader, len(trainset), len(testset)
