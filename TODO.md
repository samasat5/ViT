# TODO

1. Attention : 
    - Best practice : pré-calculer la partie géométrique (distances) une seule fois et la stocker comme buffer :
    register_buffer("distances", ...) (et ça suit automatiquement .to(device) !)

2. Train : 
    - Si tu utilises pin_memory=True dans DataLoader, tu peux accélérer les copies CPU->GPU :
    train_loader = DataLoader(..., pin_memory=True)
    ...
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

3. Normalisation CIFAR: **DONE**
    - Pour CIFAR-10, les stats classiques sont :
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

4. Transformer: **DONE**
    - PRR

5. Encoder:
    - Droppath
    - dpr in Transformer

6. VisionTransformer:
    - revoir forward

### locat + classif = OK
### classif = OK
### locat + seg = OK
### seg = OK