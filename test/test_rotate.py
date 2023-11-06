import torch
import tifffile as tiff

model = torch.load('/media/ExtHDD01/logs/Fly0B/cyc4_1024/cyc4/checkpoints/netGXY_model_epoch_100.pth', map_location='cpu').cuda()

o = tiff.imread('/media/ghc/GHc_data2/BRC/RotateImage/xyzori.tif')
#o[o > 2000] = 2000
o = o / o.max()
o = (o - 0.5) * 2

f = tiff.imread('/media/ghc/GHc_data2/BRC/RotateImage/xyzft0.tif')
f[f > 5] = 5
f = f / f.max()
f = (f - 0.5) * 2

o = torch.from_numpy(o)
f = torch.from_numpy(f)

o = o.type(torch.FloatTensor)
f = f.type(torch.FloatTensor)

i = 800
input = (torch.cat([o[:, i, :].unsqueeze(0), f[:, i, :].unsqueeze(0)], 0)).unsqueeze(0).cuda()
out = model(input)


tiff.imwrite('/media/ghc/GHc_data2/BRC/RotateImage/in.tif', o[:, i, :].numpy())
tiff.imwrite('/media/ghc/GHc_data2/BRC/RotateImage/out.tif', out['out1'][0, 0, ::].detach().cpu().numpy())
