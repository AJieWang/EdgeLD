from VGG.mydefine_VGG13 import VGG_model, COMPUTE_CONV_BLOCKS, COMPUTE_CONV_BLOCKS_PABC, COMPUTE_PARTIAL_BLOCKS, COMPUTE_BLOCK_1_6
datanode_num_temp = 3
class Num_set_up(object):
    def __init__(self ):
        self.namenode_num = 1
        self.datanode_num = datanode_num_temp
    def set_namenode_num(self, num):
        self.namenode_num = num
    def get_namenode_num(self):
        self.set_namenode_num(1)
        return self.namenode_num
    def set_datanode_num(self, num):
        self.datanode_num = num
    def get_datanode_num(self):
        self.set_datanode_num(datanode_num_temp)
        return self.datanode_num
