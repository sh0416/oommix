import sys
import logging
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from data import create_test_dataset
from data import CollateFn
from model import create_model


import numpy as n, pylab as p, time

def _angle_to_point(point, centre):
    '''calculate angle in 2-D between points and x axis'''
    delta = point - centre
    res = n.arctan(delta[1] / delta[0])
    if delta[0] < 0:
        res += n.pi
    return res

def _draw_triangle(p1, p2, p3, **kwargs):
    tmp = n.vstack((p1,p2,p3))
    x,y = [x[0] for x in zip(tmp.transpose())]
    p.fill(x,y, **kwargs)

def area_of_triangle(p1, p2, p3):
    '''calculate area of any triangle given co-ordinates of the corners'''
    return n.linalg.norm(n.cross((p2 - p1), (p3 - p1)))/2.


def convex_hull(points, graphic=False, smidgen=0.0075):
    '''
    Calculate subset of points that make a convex hull around points
    Recursively eliminates points that lie inside two neighbouring points until only convex hull is remaining.

    :Parameters:
    points : ndarray (2 x m)
    array of points for which to find hull
    graphic : bool
    use pylab to show progress?
    smidgen : float
    offset for graphic number labels - useful values depend on your data range

    :Returns:
    hull_points : ndarray (2 x n)
    convex hull surrounding points
    '''

    if graphic:
        p.clf()
        p.plot(points[0], points[1], 'ro')
    n_pts = points.shape[1]
    assert(n_pts > 5)
    centre = points.mean(1)
    if graphic: p.plot((centre[0],),(centre[1],),'bo')
    angles = n.apply_along_axis(_angle_to_point, 0, points, centre)
    pts_ord = points[:,angles.argsort()]
    if graphic:
        for i in xrange(n_pts):
            p.text(pts_ord[0,i] + smidgen, pts_ord[1,i] + smidgen, \
                   '%d' % i)
    pts = [x[0] for x in zip(pts_ord.transpose())]
    prev_pts = len(pts) + 1
    k = 0
    while prev_pts > n_pts:
        prev_pts = n_pts
        n_pts = len(pts)
        if graphic: p.gca().patches = []
        i = -2
        while i < (n_pts - 2):
            Aij = area_of_triangle(centre, pts[i],     pts[(i + 1) % n_pts])
            Ajk = area_of_triangle(centre, pts[(i + 1) % n_pts], \
                                   pts[(i + 2) % n_pts])
            Aik = area_of_triangle(centre, pts[i],     pts[(i + 2) % n_pts])
            if graphic:
                _draw_triangle(centre, pts[i], pts[(i + 1) % n_pts], \
                               facecolor='blue', alpha = 0.2)
                _draw_triangle(centre, pts[(i + 1) % n_pts], \
                               pts[(i + 2) % n_pts], \
                               facecolor='green', alpha = 0.2)
                _draw_triangle(centre, pts[i], pts[(i + 2) % n_pts], \
                               facecolor='red', alpha = 0.2)
            if Aij + Ajk < Aik:
                if graphic: p.plot((pts[i + 1][0],),(pts[i + 1][1],),'go')
                del pts[i+1]
            i += 1
            n_pts = len(pts)
        k += 1
    return n.asarray(pts)

def generate_convex_hull_point_list(points):
    convex = convex_hull(points)
    x, y = [], []
    for point in convex:
        x.append(point[0])
        y.append(point[1])
    x.append(convex[0][0])
    y.append(convex[0][1])
    return n.array(x), n.array(y)


if __name__ == "__main__":

    import scipy.interpolate as interpolate

#    fig = p.figure(figsize=(10,10))

    theta = 2*n.pi*n.random.rand(1000)
    r = n.random.rand(1000)**0.5
    x,y = r*p.cos(theta),r*p.sin(theta)

    points = n.ndarray((2,len(x)))
    points[0,:], points[1,:] = x, y

    x, y = generate_convex_hull_point_list(points)

    #Taken from https://stackoverflow.com/questions/14344099/numpy-scipy-smooth-spline-representation-of-an-arbitrary-contour-flength
    dist = np.sqrt((x[:-1] - x[1:])**2 + (y[:-1] - y[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    spline, u = interpolate.splprep([x, y], u=dist_along, s=0)

    interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    interp_x, interp_y = interpolate.splev(interp_d, spline)
    #plot(interp_x, interp_y, '-o')
    p.plot(interp_x, interp_y, '-')

    p.savefig("test.png")
    p.show()

assert False

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().cpu().numpy()
    return hook


def probe(model, loader, device):
    np.set_printoptions(precision=4, linewidth=1e+8, suppress=True)
    df = pd.DataFrame()
    activations = defaultdict(list)

    with torch.no_grad():
        model.eval()
        correct, count = 0, 0
        for batch in loader:
            input_ids = batch["inputs"]["input_ids"].to(device) 
            attention_mask = batch["inputs"]["attention_mask"].to(device) 
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = outputs.argmax(dim=1)
            # Create rows for this batch
            rows = {
                "context_idx": [i.item() for i in batch["idx"] for _ in range(batch["inputs"]["input_ids"].shape[1])],
                "token_idx": [x.item() for l in batch["inputs"]["input_ids"] for x in l],
            }
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            for name, t in activation.items():
                activations[name].append(np.reshape(t, (-1, t.shape[2])))

    df.to_csv("representations.csv", index=False)
    for name, l in activations.items():
        np.savetxt(name+'.csv', np.vstack(l), fmt="%.4f", delimiter=",", encoding='utf8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reproducibility parameter
    parser.add_argument("--seed", type=int, default=0)
    # Data hyperparameter
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["ag_news", "yahoo_answer"], default="ag_news")
    # Model hyperparameter
    parser.add_argument("--restore", type=str, required=True)
    # Train hyperparameter - augmentation
    parser.add_argument("--augment", type=str, choices=["none", "tmix", "adamix", "proposed"], default="none")
    parser.add_argument("--mixup_layer", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--coeff_intr", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    test_dataset = create_test_dataset(dataset=args.dataset,
                                       dirpath=args.data_dir,
                                       tokenizer=tokenizer)
    collate_fn = CollateFn(tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(augment=args.augment, mixup_layer=args.mixup_layer,
                         n_class=test_dataset.n_class, n_layer=12, drop_prob=0.0)
    model.to(device)
    model.load_state_dict(torch.load(args.restore))
    for name, module in model.mix_model.embedding_model.named_modules():
        if name in ["embedding_norm",
                    "encoder.0.norm", "encoder.0.ff_norm", "encoder.1.norm", "encoder.1.ff_norm",
                    "encoder.2.norm", "encoder.2.ff_norm", "encoder.3.norm", "encoder.3.ff_norm",
                    "encoder.4.norm", "encoder.4.ff_norm", "encoder.5.norm", "encoder.5.ff_norm",
                    "encoder.6.norm", "encoder.6.ff_norm", "encoder.7.norm", "encoder.7.ff_norm",
                    "encoder.8.norm", "encoder.8.ff_norm", "encoder.9.norm", "encoder.9.ff_norm",
                    "encoder.10.norm", "encoder.10.ff_norm", "encoder.11.norm", "encoder.11.ff_norm"]:
            module.register_forward_hook(get_activation(name))

    probe(model, test_loader, device)
