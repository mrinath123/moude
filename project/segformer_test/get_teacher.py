import torch

pretrained_weights = torch.load('/BS/DApt/work/project/segformer_test/work_dirs/night_convp/best_mIoU_iter_2500.pth')
#xxx = torch.load('/BS/DApt/work/project/segformer_test/work_dirs/b5_night_IDASS_inc_v2/teacher.pth')

new_dict = {}
for n,px in pretrained_weights['state_dict'].items():
    #print(n)
    parts = n.split('.')
    if 'backbone' in n:
        new_n = 'backbone' + '.' + '.'.join(parts[1:])
        new_dict[new_n] = px

    if 'decode_head' in n:
        new_n = 'decode_head' + '.' + '.'.join(parts[1:])
        new_dict[new_n] = px

torch.save(new_dict, '/BS/DApt/work/project/segformer_test/work_dirs/night_convp/student.pth')

new_dict = {}
for n,px in pretrained_weights['state_dict'].items():
    #print(n)
    parts = n.split('.')
    if 'teacher_backb' in n:
        new_n = 'backbone' + '.' + '.'.join(parts[1:])
        new_dict[new_n] = px

    if 'teacher_dec' in n:
        new_n = 'decode_head' + '.' + '.'.join(parts[1:])
        new_dict[new_n] = px
        

torch.save(new_dict, '/BS/DApt/work/project/segformer_test/work_dirs/night_convp/teacher.pth')