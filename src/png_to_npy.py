import argparse
import glob
import imageio
import os
import joblib

import numpy as np


def dir_to_npy(dir, save=False, outfile=None):
    """
    This will convert a directory of png images to a single npy file
    Parameters:
    dir: string, directory containing png files
    save: boolean, if true this module will save the npy file directly
    outfile: string, directory to save npy file if save == true
    """
    vid = []
    for im in sorted(glob.glob(dir+'/*')):
        img = imageio.imread(im)
        vid.append(img)
    if save:
        key = dir.split(os.path.sep)[-1]
        fname = "{}.npy".format(key)
        outfile = os.path.join(args['output'], fname)
        np.save(outfile, np.array(vid))
        return(None)
    return(np.array(vid))


if __name__ == "__main__":
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(
        description=('Reads all png files from video subdirectories and ',
                     'creates an npy file for each subdirectory'),
        add_help='How to use', prog='png_to_npy.py <args>')

    # Required arguments
    parser.add_argument("-i", "--input", required=True,
                        help=("path to the directory containing video ",
                              "subdirectories"))

    # optional arguments
    parser.add_argument("-o", "--output", default=os.path.join(cwd, "videos"),
                        help=("The destination to store all of the npy files ",
                              "[default: cwd/videos]"))
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help=('degree of parallelism to use -1 will use all ',
                              'present cores[default: -1]'))
    parser.add_argument("-s", "--save_inproc", action="store_true",
                        help=("if true data will be written in time (good for",
                              " large datasets)"))

    # parse input and output arguments
    args = vars(parser.parse_args())
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    file_list = glob.glob(args['input']+'*')

    # run over all input in parallel
    out = joblib.Parallel(n_jobs=args['n_jobs'], verbose=10,)(
          joblib.delayed(dir_to_npy)(f, args['save_inproc'], args['output'])
          for f in file_list
    )

    # if save is set to save at the end, save all data at once
    if not args["save_inproc"]:
        for outs, f in zip(out, file_list):
            key = f.split(os.path.sep)[-1].split(".")[0]
            fname = "{}.npy".format(key)
            outfile = os.path.join(args['output'], fname)
            np.save(outfile, outs)
