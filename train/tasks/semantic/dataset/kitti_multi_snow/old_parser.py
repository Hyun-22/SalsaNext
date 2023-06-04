import numpy as np
import torch, time
import glob, os, sys
from torch.utils.data import Dataset
import random

from common.laserscan import LaserScan, SemLaserScan

def load_label_file(label_file):
    label = np.fromfile(label_file, dtype=np.uint32)
    label = label.reshape((-1))
    label = label & 0xFFFF   # get lower half for semantics
    return label

def merge_batch(batch_list):
    stacked_points = np.zeros((0, 7))
    stacked_coordinates = np.zeros((0, 4))    
    stacked_label = []
    stacked_seqs = []
    stacked_names = []
    stacked_lga_idx = []
    stacked_back_idx = []
    
    stacked_len_point_list = []
    stacked_weather_label = []
    for i, (points, coordinates, labels, seq, file_name, after_idx, lga_full_idx, weather_label) in enumerate(batch_list):
        stacked_points = np.concatenate((stacked_points, points), axis = 0)

        # make coordinates with batch numbering
        coord_pad = np.pad(coordinates, ((0, 0), (1, 0)), mode='constant', constant_values=i)
        stacked_coordinates = np.concatenate((stacked_coordinates, coord_pad), axis = 0)
        stacked_label = np.concatenate((stacked_label, labels), axis=0)        
        stacked_lga_idx = np.concatenate((stacked_lga_idx, lga_full_idx), axis=0)
        
        stacked_seqs = stacked_seqs + [seq]
        stacked_names = stacked_names + [file_name]
        
        stacked_len_point_list.append(len(points))
        stacked_weather_label.append(weather_label)
    # print(stacked_points.shape)
    stacked_len_point_list = np.array(stacked_len_point_list)
    stacked_weather_label = np.array(stacked_weather_label)
    return stacked_points, stacked_coordinates, stacked_label, stacked_seqs, stacked_names, stacked_lga_idx, stacked_len_point_list, stacked_weather_label

def my_collate(batch):
    data = [item[0] for item in batch]
    project_mask = [item[1] for item in batch]
    proj_labels = [item[2] for item in batch]
    data = torch.stack(data,dim=0)
    project_mask = torch.stack(project_mask,dim=0)
    proj_labels = torch.stack(proj_labels, dim=0)

    to_augment =(proj_labels == 12).nonzero()
    to_augment_unique_12 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 5).nonzero()
    to_augment_unique_5 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 8).nonzero()
    to_augment_unique_8 = torch.unique(to_augment[:, 0])

    to_augment_unique = torch.cat((to_augment_unique_5,to_augment_unique_8,to_augment_unique_12),dim=0)
    to_augment_unique = torch.unique(to_augment_unique)

    for k in to_augment_unique:
        data = torch.cat((data,torch.flip(data[k.item()], [2]).unsqueeze(0)),dim=0)
        proj_labels = torch.cat((proj_labels,torch.flip(proj_labels[k.item()], [1]).unsqueeze(0)),dim=0)
        project_mask = torch.cat((project_mask,torch.flip(project_mask[k.item()], [1]).unsqueeze(0)),dim=0)

    return data, project_mask,proj_labels
  
class SemanticKitti(Dataset):
    def __init__(self, db_type, data_path, config, label_map, seqs, 
                 sensor,              # sensor to parse scans from
               max_points=150000,   # max number of points present in dataset
               gt=True,transform = False):
        'Initialization'
        # self.data_paths = os.path.join(data_path, "sequences")
        if isinstance(data_path, list):
          self.data_paths = [os.path.join(data, "sequences") for data in data_path]
        if isinstance(data_path, str):
           self.data_paths = [os.path.join(data_path, "sequences")]
        print("self.data_paths")
        print(self.data_paths)
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"],
                                            dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                            dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points
        self.gt = gt
        self.transform = transform
        self.config = config
        self.has_label = False
        # self.device = device
                
        if db_type == "train":
            self.has_label = True
            self.rotate_aug = True
            self.flip_aug = True
                    
        elif db_type == "valid_mix":
            self.has_label = True
            self.rotate_aug = False
            self.flip_aug = False
            
        elif db_type == "valid_normal":
            self.has_label = True
            self.rotate_aug = False
            self.flip_aug = False
            
        else:
            self.has_label = False
            self.rotate_aug = False
            self.flip_aug = False

        self.pcd_file = []
        self.label_files = []
        
        if db_type == "train":
            for idx, data_path in enumerate(self.data_paths):
                tmp_pcd_files = []
                tmp_label_files = []
                for seq in seqs:
                    seq_path = os.path.join(data_path, seq)
                    for (path, dir, files) in os.walk(seq_path + "/velodyne"):
                        for filename in files:
                            file, ext = os.path.splitext(filename)
                            if ext == '.bin':
                                tmp_pcd_files.append(os.path.join(path, filename))
                            if self.has_label == True:
                                tmp_label_files.append(os.path.join(seq_path,"labels", "{}.label".format(file)))
                # tmp_pcd_files.sort()
                # tmp_label_files.sort() 
                crop_len = len(tmp_pcd_files) // len(self.data_paths)
                self.pcd_file += tmp_pcd_files[crop_len * idx:crop_len * (idx + 1)]
                self.label_files += tmp_label_files[crop_len * idx:crop_len * (idx + 1)]
                tmp_pcd_files.clear()
                tmp_label_files.clear()
                print(len(self.pcd_file))
            
            # self.pcd_file = self.pcd_file[:30]
            # self.label_files = self.label_files[:30]

        elif db_type == "valid_mix":
            for idx, data_path in enumerate(self.data_paths):
                tmp_pcd_files = []
                tmp_label_files = []
                for seq in seqs:
                    seq_path = os.path.join(data_path, seq)
                    for (path, dir, files) in os.walk(seq_path + "/velodyne"):
                        for filename in files:
                            file, ext = os.path.splitext(filename)
                            if ext == '.bin':
                                tmp_pcd_files.append(os.path.join(path, filename))
                            if self.has_label == True:
                                tmp_label_files.append(os.path.join(seq_path,"labels", "{}.label".format(file)))

                crop_len = len(tmp_pcd_files) // len(self.data_paths)
                self.pcd_file += tmp_pcd_files[crop_len * idx:crop_len * (idx + 1)]
                self.label_files += tmp_label_files[crop_len * idx:crop_len * (idx + 1)]
                tmp_pcd_files.clear()
                tmp_label_files.clear()
                print(len(self.pcd_file))
            print("len in snow 0 : {}".format(len(self.pcd_file)))
            # print("validation sample")
            # print(self.pcd_file[:10])
            # print(self.label_files[:10])
            
        elif db_type == "valid_normal":
            for seq in seqs:
                seq_path = os.path.join(self.data_paths[-1], seq)
                for (path, dir, files) in os.walk(seq_path + "/velodyne"):
                    for filename in files:
                        file, ext = os.path.splitext(filename)
                        if ext == '.bin':
                            self.pcd_file.append(os.path.join(path, filename))
                        if self.has_label == True:
                            self.label_files.append(os.path.join(seq_path,"labels", "{}.label".format(file)))
        
            print(len(self.pcd_file))

        self.len = len(self.pcd_file)
        self.label_map = label_map
        self.db_type = db_type

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.pcd_file)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # scan = np.fromfile(self.pcd_file[index], dtype=np.float32)
        # absolute_xyz = scan.reshape((-1, 4))
        # sequences, pcd_file_name = os.path.split(self.pcd_file[index])
        # file_name = os.path.splitext(pcd_file_name)[0]
        # seq = os.path.split(sequences)[0]
        scan_file = self.pcd_file[index]
        if self.gt:
            label_file = self.label_files[index]
        if self.has_label:
            labels = load_label_file(self.label_files[index])
            # extract label information from label filename
            snowrain_mm = self.label_files[index].split(os.sep)[-5]
            if snowrain_mm in ["snow_30", "rain_30"]:
              weather_label = 3
            elif snowrain_mm in ["snow_20", "rain_20"]:
              weather_label = 2
            elif snowrain_mm in ["snow_10", "rain_10"]:
              weather_label = 1
            else:
              weather_label = 0
            labels = self.map(labels, self.label_map)             
        else:
            # labels = np.zeros(len(absolute_xyz))
            weather_label = 0
            # open a semantic laserscan
        DA = False
        flip_sign = False
        rot = False
        drop_points = False
        if self.transform:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    DA = True
                if random.random() > 0.5:
                    flip_sign = True
                if random.random() > 0.5:
                    rot = True
                drop_points = random.uniform(0, 0.5)

        if self.gt:
            scan = SemLaserScan(self.color_map,
                                project=True,
                                H=self.sensor_img_H,
                                W=self.sensor_img_W,
                                fov_up=self.sensor_fov_up,
                                fov_down=self.sensor_fov_down,
                                DA=DA,
                                flip_sign=flip_sign,
                                drop_points=drop_points)
        else:
            scan = LaserScan(project=True,
                            H=self.sensor_img_H,
                            W=self.sensor_img_W,
                            fov_up=self.sensor_fov_up,
                            fov_down=self.sensor_fov_down,
                            DA=DA,
                            rot=rot,
                            flip_sign=flip_sign,
                            drop_points=drop_points)

        # open and obtain scan
        scan.open_scan(scan_file)
        if self.gt:
            scan.open_label(label_file)
        # map unused classes to used classes (also for projection)
        scan.sem_label = self.map(scan.sem_label, self.learning_map)
        scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        if self.gt:
            unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
            unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
        else:
            proj_labels = []
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                        proj_xyz.clone().permute(2, 0, 1),
                        proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        # get name and sequence
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")
        return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points
        # return points, coordinates, labels, seq, file_name, after_idx, lga_full_idx, weather_label

    # augmentation function (rotation)
    def augmentation_rotate(self, xyz):
      # random rotate augmentation
      rotate_rad = np.deg2rad(np.random.random() * 360)
      c, s = np.cos(rotate_rad), np.sin(rotate_rad)
      j = np.matrix([[c, s], [-s, c]])
      xyz[:, :2] = np.dot(xyz[:, :2], j)
      return xyz
      
    # augmentation function (flip)
    def augmentation_flip(self, xyz):
      flip_type = np.random.choice(4, 1)
      if flip_type == 1:
          xyz[:, 0] = -xyz[:, 0]
      elif flip_type == 2:
          xyz[:, 1] = -xyz[:, 1]
      elif flip_type == 3:
          xyz[:, :2] = -xyz[:, :2]
      return xyz
  
    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
          if isinstance(data, list):
            nel = len(data)
          else:
            nel = 1
          if key > maxkey:
            maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
          lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
          lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
          try:
            lut[key] = data
          except IndexError:
            print("Wrong key ", key)
        # do the mapping
        return lut[label]

class Parser():
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True,  # shuffle training set?
               ):
    super(Parser, self).__init__()

    # if I am training, get the dataset
    self.root = root
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.batch_size = batch_size
    self.workers = workers
    self.gt = gt
    self.shuffle_train = shuffle_train
    self.collate_func = merge_batch
    self.max_points = max_points

    # number of classes that matters is the one for xentropy
    self.nclasses = len(self.learning_map_inv)

    # Data loading code
    print("\n------------------------------------------------------------------------------")
    print("Train sequence: ", self.train_sequences)
    self.train_dataset = SemanticKitti(db_type="train",
                                     data_path = self.root,
                                     config = self.voxel_config,
                                     label_map = self.learning_map,
                                     seqs = self.train_sequences)
    self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=self.shuffle_train,
                                                   num_workers=self.workers,
                                                   collate_fn=self.collate_func,
                                                   pin_memory=True,
                                                   drop_last=True)
    self.trainiter = iter(self.trainloader)

    print("Valid sequence: ", self.valid_sequences)
    self.valid_dataset = SemanticKitti("valid_mix",
                                     self.root,
                                     self.voxel_config,
                                     self.learning_map,
                                     self.valid_sequences)
    # self.valid_dataset = SemanticKitti("valid_normal",
    #                                  self.root,
    #                                  self.voxel_config,
    #                                  self.learning_map,
    #                                  self.valid_sequences)
    self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   collate_fn=self.collate_func,
                                                   pin_memory=True,
                                                   drop_last=True)
    
    self.validiter = iter(self.validloader)

    # if self.test_sequences:
    print("Test sequence: ", self.test_sequences)
    self.test_dataset = SemanticKitti("test",
                                      self.root,
                                      self.voxel_config,
                                      self.learning_map,
                                      self.test_sequences)
    self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=self.workers,
                                                  collate_fn=self.collate_func,                                                    
                                                  pin_memory=True,
                                                  drop_last=True)
    self.testiter = iter(self.testloader)
    print("------------------------------------------------------------------------------\n")

  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return SemanticKitti.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.learning_map)

  def to_color(self, label):
    # put label in original values
    label = SemanticKitti.map(label, self.learning_map_inv)
    # put label in color
    return SemanticKitti.map(label, self.color_map)