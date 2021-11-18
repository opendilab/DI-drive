import os
import json
import lmdb
import shutil
import numpy as np


def write_json(file_path, json_data):
    with open(os.path.join(file_path), 'w') as fo:
        json_obj = {}
        for key, value in json_data.items():
            json_obj.update({key: value})
        fo.write(json.dumps(json_obj, sort_keys=True, indent=4))


def write_lmdb(file_path, lmdb_data):
    lmdb_env = lmdb.open(file_path, map_size=1e10)

    with lmdb_env.begin(write=True) as txn:
        for key, value in lmdb_data.items():
            txn.put(key=key.encode(), value=value)


def write_episode_lmdb(episode_path, episode_data, lmdb_obs_type=None):
    lmdb_env = lmdb.open(os.path.join(episode_path, "measurements.lmdb"), map_size=1e10)
    with lmdb_env.begin(write=True) as txn:
        txn.put('len'.encode(), str(len(episode_data)).encode())
        for i, x in enumerate(episode_data):
            measurements = x[0]
            txn.put(('measurements_%05d' % i).encode(), np.ascontiguousarray(measurements).astype(np.float32))
            sensor_data = x[1]
            for key in sensor_data:
                if lmdb_obs_type and key in lmdb_obs_type:
                    txn.put(('%s_%05d' % (key, i)).encode(), np.ascontiguousarray(sensor_data[key].astype(np.uint8)))
            others = x[2]
            for key in others.keys():
                txn.put(('%s_%05d' % (key, i)).encode(), np.ascontiguousarray(others[key]).astype(np.float32))
