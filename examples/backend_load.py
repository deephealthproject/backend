#!/usr/bin/env python3

# Copyright (c) 2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Authors:
#  * Francesco Versaci <cesco@crs4.it>
#  * Luca Pireddu <pireddu@crs4.it>

import argparse
import time

import requests


def backend_load_test(args):
    base_url = args.backend_url

    infer_payload =  {
        "modelweights_id": args.modelweights_id,
        "dataset_id": args.dataset_id,
        "project_id": args.project_id
    }

    procs = []
    start = time.perf_counter()
    # start jobs
    for _ in range(args.n_jobs):
        r = requests.post(base_url+'inference', data=infer_payload)
        if r.status_code != 201:
            print(r.text)
            raise RuntimeError(r.text)
        body = r.json()
        print(body)
        proc_id = body['process_id']
        procs.append(proc_id)
    # wait for jobs to finish
    time.sleep(30)
    finished = False
    while not finished:
        finished = True
        for proc_id in procs:
            r = requests.get(base_url+'status', params={'process_id':proc_id})
            fin = (r.json()['status']['process_status'] == 'finished')
            finished &= fin
            # if (fin):
            #     print(f"{proc_id}: {r.json()['status']['process_data']} -- Finished")
            # else:
            #     print(f"{proc_id}: {r.json()['status']['process_data']}")
        # print('Waiting and retrying...')
        time.sleep(30)
    # measure and print duration
    dur = time.perf_counter() - start
    # print(f'---> Load: {load_per_node} Duration: {dur/60:.4} minutes')
    return dur


def create_parser():
    parser = argparse.ArgumentParser(description="Run a load on the DeepHealth back end")

    def pos_type(x):
        x = int(x)
        if x <= 0:
            raise argparse.ArgumentTypeError("value must be greater than 0")
        return x

    parser.add_argument("--dataset-id", type=pos_type, required=True)
    parser.add_argument('--project-id', type=pos_type, required=True)
    parser.add_argument('--modelweights-id', type=pos_type, required=True)
    parser.add_argument('--n-jobs', type=pos_type, default=10,
                        help="Number of job copies to create")
    parser.add_argument('backend-url')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    backend_load_test(args)

if __name__ == '__main__':
    main()
