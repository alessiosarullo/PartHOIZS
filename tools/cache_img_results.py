import sys

import h5py
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms
from torchvision.models import resnet152

from config import cfg
from lib.dataset.hicodet_hake import HicoDetHakeSplit
from lib.dataset.vcoco import VCocoSplit



def forward(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    img_feats = x.view(x.size(0), -1)
    cl_unbounded_scores = model.fc(img_feats)

    return img_feats, cl_unbounded_scores


def save_feats():
    # sys.argv += []  # TODO choice of HICO or VCOCO in args
    cfg.parse_args(fail_if_missing=False)

    if cfg.debug:
        try:  # PyCharm debugging
            print('Starting remote debugging (resume from debug server)')
            import pydevd_pycharm
            pydevd_pycharm.settrace('130.88.195.105', port=16004, stdoutToServer=True, stderrToServer=True)
            print('Remote debugging activated.')
        except:
            print('Remote debugging failed.')
            raise

    if False:
        splits = HicoDetHakeSplit.get_splits()
    else:
        splits = VCocoSplit.get_splits()

    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    vm = resnet152(pretrained=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        vm.cuda()
    else:
        device = torch.device('cpu')
        print('!!!!!!!!!!!!!!!!! Running on CPU!')
    vm.eval()
    for split in ['train', 'test']:
        hds = splits[split]

        precomputed_feats_fn = cfg.precomputed_feats_format % ('new_cached_file', 'resnet152', split)
        with h5py.File(precomputed_feats_fn, 'w') as feat_file:

            all_img_feats, all_cl_unbounded_scores = [], []
            num_imgs = len(hds)
            for img_id in range(num_imgs):
                img = hds.get_img(img_id)
                img = img_transform(img).to(device=device)
                img_feats, cl_unbounded_scores = forward(vm, img.unsqueeze(dim=0))
                all_img_feats.append(img_feats.detach().cpu().numpy())
                all_cl_unbounded_scores.append(cl_unbounded_scores.detach().cpu().numpy())
                if img_id % 10 == 0 or img_id == num_imgs - 1:
                    print('Image %6d/%d' % (img_id, num_imgs))
                torch.cuda.empty_cache()

            all_img_feats = np.concatenate(all_img_feats, axis=0)
            assert all_img_feats.shape[0] == num_imgs

            feat_file.create_dataset('img_feats', data=all_img_feats)
            feat_file.create_dataset('scores', data=all_cl_unbounded_scores)

    print(f'Done. Remember to rename the output file! Current name is {precomputed_feats_fn}.')


if __name__ == '__main__':
    save_feats()
