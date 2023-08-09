import random


class PBRS():
    def __init__(self, capacity, num_classes):
        self.data = [[] for _ in range(num_classes)]
        self.counter = [0] * num_classes
        self.capacity = capacity
        self.num_classes = num_classes

    def get_memory(self):

        data = self.data

        tmp_data = []
        for images_per_cls in data:
            tmp_data.extend(images_per_cls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for images_per_cls in self.data:
            occupancy += len(images_per_cls)
        return occupancy

    def add_instance(self, image, cls):
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            self.data[cls].append(image)

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(0, len(self.data[largest]))  # target index to remove
            self.data[largest].pop(tgt_idx)
        else:# replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(0, len(self.data[cls]))  # target index to remove
                self.data[cls].pop(tgt_idx)
            else:
                return False
        return True
    
    def get_largest_indices(self):
        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices
    
    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_classes
        for i, images_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(images_per_cls)
        return occupancy_per_class