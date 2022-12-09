from torch.utils.data import Dataset
import utils

img_dir = '224x224';

class ImageDataset(Dataset):
    def __init__(self, csv_path, transform):
        images, labels = utils.load_data(csv_path)

        self.images = images
        self.labels = labels
        self.transform = transform
        self.classes = ('Arts & Photography', 'Biographies & Memoirs', 'Business & Money', 'Calendars',
                        "Children's Books", 'Comics & Graphic Novels', 'Computers & Technology',
                        'Cookbooks, Food & Wine', 'Crafts, Hobbies & Home', 'Christian Books & Bibles',
                        'Engineering & Transportation', 'Health, Fitness & Dieting', 'History', 'Humor & Entertainment',
                        'Law', 'Literature & Fiction', 'Medical Books', 'Mystery, Thriller & Suspense', 'Parenting & Relationships',
                        'Politics & Social Sciences', 'Reference', 'Religion & Spirituality', 'Romance', 'Science & Math',
                        'Science Fiction & Fantasy', 'Self-Help', 'Sports & Outdoors', 'Teen & Young Adult', 'Test Preparation'
                        'Travel')

        assert len(self.images) == len(self.labels)

    def __len__(self):
        """Returns the number of examples in the dataset"""
        return len(self.labels)
    
    def num_classes(self):
        """Returns the number of classes in the dataset"""
        return len(self.classes)
    
    def get_class(self, label):
        """Returns the name of the bird corresponding to the given label value"""
        return self.classes[label]

    def get_image(self, idx):
        """Returns the image of the idx'th example in the dataset"""
        return self.images[idx]

    def get_label(self, idx):
        """Returns the label of the idx'th example in the dataset"""
        return self.labels[idx]
    
    def __getitem__(self, idx):
        """Returns a tuple of the *transformed* image and label of the idx'th example in the dataset"""
        return (self.transform(self.images[idx]), self.labels[idx])
    
    def display(self, idx):
        """Displays the image at a given index"""
        display(self.get_image(idx))

