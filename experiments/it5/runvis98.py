import os

# specify the range of features that should be visualized for the -l layer specified below, e.g. (-l 15)

start_feature = 1
end_feature = 4095

# end_feature ViT-B = 3071
# end_feature ViT-L = 4095

# ViT-B: -l 11 = final layer
# ViT-L: -l 23 = final layer

# loop over all features
for feature in range(start_feature, end_feature + 1):
    # construct the command
    command = f"python vis98.py -l 15 -f {feature} -n 98"
    
    # execute the command
    os.system(command)

# -n = Network:
#
# 94:     CLIP0_RN50x4
# 95:     CLIP1_RN50x16
# 96:     CLIP2_ViT-B/32
# 97:     CLIP3_ViT-B/16
# 98:     CLIP4_ViT-L/14
# 99:     CLIP5_ViT-L/14@336px