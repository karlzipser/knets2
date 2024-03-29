
from ..imports import *


class Multi_imagefolder_dataset(torch.utils.data.Dataset):

    def __init__(
        _,
        root,
        geometric_transforms,
        color_transforms,
        usefx=True,
        asrgb255=False,
        proportion_of_data_to_use=1.,
        e=0,
        include_untransformed_copy_of_img=False,
    ):
        super(Multi_imagefolder_dataset,_).__init__()
        _.e=e
        _.include_untransformed_copy_of_img=include_untransformed_copy_of_img
        _.asrgb255=asrgb255
        if isNone(geometric_transforms):
            _.geometric_transforms=_._identity
        else:
            _.geometric_transforms=geometric_transforms
        if isNone(color_transforms):
            _.color_transforms=_._identity
        else:
            _.color_transforms=color_transforms
        # expect input and output folders to have images of the same names
        # so that they get loaded in register
        imgfolderpaths=sggo(root,'*')
        _.imgpathsdict={}
        for p in imgfolderpaths:
            _.imgpathsdict[p]=sggo(p,'*.*') # assume only images in folders
            cb('len(',p,')=',len(_.imgpathsdict[p]),e=_.e)
        _test={}
        for p in _.imgpathsdict:
            _test[p]=[]
            for f in _.imgpathsdict[p]:
                _test[p].append(fname(f))
            _test[p]='\n'.join(_test[p])
        ks=kys(_test)
        k0=ks[0]
        for i in range(1,len(ks)):
            k=ks[i]
            assert _test[k0]==_test[k]
        _.usefx=usefx
        _.normalize_transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        _.totensor_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        _.proportion_of_data_to_use=proportion_of_data_to_use
        #cE('__init__',proportion_of_data_to_use)
    def _identity(_,x):
        return x

    def __getitem__(_, index):

        if False:#_.proportion_of_data_to_use<1:
            ln=_.__len__()
            maxindex=int(_.proportion_of_data_to_use*ln)
            _i=int(maxindex*index/ln)
            cb('\tindex:',index,'goes to',_i)
            index=_i
        seed = np.random.randint(2147483647)
        imgs=[]
        if _.include_untransformed_copy_of_img:
            r=[0,1]
        else:
            r=[1]
        for p in _.imgpathsdict:
            for i in r:
                f=_.imgpathsdict[p][index]
                cb(f,e=_.e)
                x=imread(f)
                if _.usefx:
                    x=fx(x)
                random.seed(seed) 
                torch.manual_seed(seed)
                x=_.totensor_transform(x)
                #cg(i,x.size())
                if i:
                    x=_.geometric_transforms(x)
                    if 'mask' not in f:
                        x=_.color_transforms(x)
                        cb('\tcolor_transforms',e=_.e)
                    else:
                        cb('\tNO color_transforms',e=_.e)
                if _.asrgb255:
                    x=cuda_to_rgb_image(x).astype(u8)
                    cb('\tasrgb255',e=_.e)
                    assert False # just to check for this
                if not _.asrgb255:
                    x=_.normalize_transform(x)
                    cb('\normalize_transform',e=_.e)
                imgs.append(x)
                #cE(i,x.size())
        imgscat=torch.cat(imgs)
        #print('imgscat',imgscat.size())
        return dict(img=imgscat,index=index,f=f)

    def __len__(_):
        ks=kys(_.imgpathsdict)
        thelen=len(_.imgpathsdict[ks[0]])
        #cE('thelen',thelen)
        return thelen




def get_transformsdict(p):

    geometric_transforms_list=[]

    if p.RandomPerspective:
        geometric_transforms_list.append(
            v2.RandomPerspective(
                distortion_scale=p.RandomPerspective_distortion_scale,
                p=p.RandomPerspective_p,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=p.RandomPerspective_fill,
            )
        )

    if p.RandomRotation:
        geometric_transforms_list.append(
            v2.RandomRotation(p.RandomRotation_angle,fill=p.RandomRotation_fill)
        )

    if p.RandomZoomOut:
        geometric_transforms_list.append(
            v2.RandomZoomOut(side_range=p.RandomZoomOut_side_range,fill=p.RandomZoomOut_fill)
        )


    if p.Pad:
        cE('p.Pad')
        geometric_transforms_list.append(
            #v2.Pad(padding=(640-360)//2,fill=p.Pad_fill)
            v2.Pad(padding=64,fill=p.Pad_fill)
        )

    if p.CenterCrop:
        geometric_transforms_list.append(
            v2.CenterCrop(size=640)
        )


    if p.RandomResizedCrop:
        geometric_transforms_list.append(
            v2.RandomResizedCrop(
                p.image_size,
                scale=p.RandomResizedCrop_scale,
                ratio=p.RandomResizedCrop_ratio,
                antialias=True,
            )
        )

    if p.Resize:
        geometric_transforms_list.append(
            v2.Resize(size=p.image_size,antialias=True)
        )

    if p.RandomHorizontalFlip:
        geometric_transforms_list.append(
            v2.RandomHorizontalFlip(p=p.RandomHorizontalFlip_p)
        )
        
    if p.RandomVerticalFlip:
        geometric_transforms_list.append(
            v2.RandomVerticalFlip(p=p.RandomVerticalFlip_p)
        )

    color_transforms_list=[]
    if p.ColorJitter:
        color_transforms_list.append(
            v2.ColorJitter(
                brightness=p.ColorJitter_brightness,
                contrast=p.ColorJitter_contrast,
                saturation=p.ColorJitter_saturation,
                hue=p.ColorJitter_hue,
            )
        )

    transformsdict=dict(
        geometric_transforms = transforms.Compose(geometric_transforms_list),
        color_transforms = transforms.Compose(color_transforms_list)
    )

    return transformsdict



def get_dataloader(p):

    transformsdict=get_transformsdict(p)

    dataloader=torch.utils.data.DataLoader(
            Multi_imagefolder_dataset(
                root=p.datapath,
                geometric_transforms=transformsdict['geometric_transforms'],
                color_transforms=transformsdict['color_transforms'],
                usefx=False,
                asrgb255=False,
                proportion_of_data_to_use=1.#,p.proportion_of_data_to_use doesn't work
            ),
            batch_size=p.batch_size,
            shuffle=True,
            num_workers=p.workers,
    )
    
    return dataloader




def get_centered_bounding_box_data(mask,show=False,img=None):
    H,W=iheight(mask),iwidth(mask)
    x,y,w,h=get_bounding_rect_from_object_mask(m1(mask))
    #print(x,y,w,h)
    yc=y+h//2-H//2
    xc=x+w//2-W//2
    overlapscenter=0
    if np.abs(xc)<w//2 and np.abs(yc)<h//2:
        overlapscenter=1
    if show:
        #assert isimg(img)
        if isNone(img):
            img=1*mask
        img2=cv2.rectangle(1*img,(x,y),(x+w,y+h),(255,0,0),2)
        sh(img2)
        plot((W//2,W//2+xc),(H//2,H//2+yc,),'bx-')
        c='k'
        if overlapscenter:
            c='r'
        plot(W//2,H//2,c+'o-')
        spause()
    return xc,yc,w,h,x,y,W,H,overlapscenter



#EOF
