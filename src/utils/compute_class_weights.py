from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
import dataset.utils as dsutils


def get_class_weights(dataloader, recompute = False):
    # check if file class_weights.txt exists
    if not recompute:
        try:
            with open('class_weights.txt') as f:
                weights = f.readlines()
                weights = [float(x.strip()) for x in weights]
                return weights
        except FileNotFoundError:
            recompute = True

    if recompute:
        print("Recomputing class weights...")
        # Define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load datasets
        
        # COCOconfig = dsutils.COCOConfig()
        # COCOconfig.transform = "RESIZE_NORMALIZE"
        # ImageNetConfig = dsutils.ImagenetConfig()
        # ImageNetConfig.transform = "RESIZE_NORMALIZE"

        # dataset_coco_train = dsutils.get_coco_dataset(mode = "train", config = COCOconfig)
        # dataset_imagenet_train = dsutils.get_imagenet_dataset(mode = "train", config = ImageNetConfig)
        # train_dataset = torch.utils.data.ConcatDataset([dataset_coco_train, dataset_imagenet_train])

        # # Define DataLoaders
        # batch_size = 64
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=15,
        #                         persistent_workers=True, pin_memory=True, prefetch_factor=2)

        train_loader = dataloader

        label_counts = Counter()
        for _, labels, _ in train_loader:
            label_counts.update(labels.numpy())  # or labels.tolist()

        total_samples = sum(label_counts.values())
        num_classes = len(label_counts)

        weights = [
            total_samples / (num_classes * label_counts[class_idx])
            for class_idx in range(num_classes)
        ]

        print(f"Class weights: {weights}")

        with open('class_weights.txt', 'w') as f:
            for item in weights:
                f.write("%f\n" % item)

        return weights
    
def get_class_dataset_weights(dataloader, recompute = False):
    # check if file class_dataset_weights.txt exists
    if not recompute:
        try:
            with open('class_dataset_weights.txt') as f:
                table = []
                for l in f.readlines():
                    row = [float(x.strip()) for x in l.split(',')]
                    table.append(row)
                return np.array(table)
        except FileNotFoundError:
            recompute = True

    if recompute:
        # Define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load datasets

        print("Recomputing class dataset weights...")
        
        # COCOconfig = dsutils.COCOConfig()
        # COCOconfig.transform = "RESIZE_NORMALIZE"
        # ImageNetConfig = dsutils.ImagenetConfig()
        # ImageNetConfig.transform = "RESIZE_NORMALIZE"

        # dataset_coco_train = dsutils.get_coco_dataset(mode = "train", config = COCOconfig)
        # dataset_imagenet_train = dsutils.get_imagenet_dataset(mode = "train", config = ImageNetConfig)
        # train_dataset = torch.utils.data.ConcatDataset([dataset_coco_train, dataset_imagenet_train])

        # # Define DataLoaders
        # batch_size = 64
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=15,
        #                         persistent_workers=True, pin_memory=True, prefetch_factor=2)

        train_loader = dataloader

        label_counts = Counter()

        # i = 0
        max_ds = 0
        for _, labels, ds in train_loader:
            max_ds = max(max(ds), max_ds)
            
            pair = np.stack([labels.numpy(), ds.numpy()], axis = -1)
            # print(f'{pair.shape=}')
            pair = [tuple(x) for x in pair]

            label_counts.update(pair)  # or labels.tolist()
            # i += 1
            # if i == 1000:
            #     break

        table = np.zeros((64, max_ds+1))
        
        for (x, y), count in label_counts.items():
            table[x, y] = count
        print(table)

        tablesum = np.sum(table, axis=1)
        tablesum = np.reshape(tablesum, (-1, 1))

        tableNew = table + tablesum*0.01 + 100
        tableNew = tablesum / tableNew
        tableNew = tableNew*(table != 0)

        print(tableNew)

        with open('class_dataset_weights.txt', 'w') as f:
            for row in tableNew:
                row = map(str, row)
                line = ','.join(row)
                f.write(line)
                f.write('\n')
            
        return tableNew

if __name__ == "__main__":
    print(get_class_weights(recompute=True))
