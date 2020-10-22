import json
import sys
import numpy as np
from pprint import pprint
import pandas as pd

def deepget(m, *ks):
    for k in ks:
        m = m[k]
    return m

def update_state(state, name, row, *path):
    try:
        state[name] = deepget(row, *path)
    except KeyError:
        return False
    return True

def load_data(rows):
    state = dict(
            system_ts=np.nan,
            scene_ts=np.nan,
            recv_ts=np.nan,
            recv_ts_mono=np.nan,
            scenario='',
            signtype='',
            x=np.nan,
            y=np.nan,
            is_visible=False,
            trial_number=np.nan,
            )
    output = []

    trial_number = 0
    def handle_row(row):
        nonlocal trial_number
        data = row['data']
        
        try:
            state['scenario'] = data['startingScenario']
            state['trial_number'] = np.nan
        except KeyError:
            pass

        if data.get('exitedScenario'):
            state['scenario'] = ''

        if data.get('discriminationTrialStart'):
            trial_number += 1
            state['trial_number'] = trial_number
        
        bd = data["pdBase"]
        state['system_ts'] = row['time']
        state['recv_ts_mono'] = row['recv_ts_mono']
        state['recv_ts'] = row['recv_ts']
        state['scene_ts'] = bd['time']
        
        state['signtype'] = bd['targetSign'] or ''
        state['is_visible'] = bd['targetVisible']
        m = bd['platform']['elements']
        state['x'] = m['12']/2.0*1080 + 1920/2
        state['y'] = m['13']/2.0*1080 + 1080/2
        

        output.append(state.copy())

    for row in rows:
        try:
            handle_row(row)
        except KeyError:
            continue
    
    data = pd.DataFrame.from_records(output)
    return data

if __name__ == '__main__':
    def rows():
        for l in sys.stdin:
            try:
                yield json.loads(l)
            except json.decoder.JSONDecodeError:
                continue
    data = load_data(rows())
    data.to_parquet(sys.stdout.buffer)
