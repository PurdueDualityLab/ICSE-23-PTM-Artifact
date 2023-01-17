from argparse import ArgumentParser as AP

def get_args():

    ap = AP()
    ap.add_argument("-m", "--models", type=str, nargs="+")
    # ap.add_argument("-e", "--eval", action="store_true")
    ap.add_argument("-i", "--input", type=str) # changed from inference
    ap.add_argument('-r','--root',type=str) # the root dir for .datasets (in progress)

    ap.add_argument('--classification',action='store_true')
    ap.add_argument('--detection',action='store_true')

    args = ap.parse_args()
    return args

