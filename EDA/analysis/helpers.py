import pandas as pd
import numpy as np
import json
import os

def get_sub_info(path, sub, ses, print_info=True):
    '''
    return and print (mono and bi) channels + channel groups, recording time, and sampling frequency info for a given subject and session.

    args:
        path (str): path to the task folder (not the subject folder)
        sub (str): subject ID
        ses (int): session number
        print_info (bool): whether to print the info or not
    
    returns:
        mono_channels (dict): dictionary of monopolar channel types and counts
        bi_channels (dict): dictionary of bipolar channel types and counts
        groups (list): list of unique channel groups
        record_dur (float): recording duration in seconds
        freq (int): sampling frequency in Hz
    '''
    # get file paths
    tasknum = path.split('/')[-1]
    prefix = f'{path}/{sub}/ses-{ses}/ieeg/{sub}_ses-{ses}_task-{tasknum}'

    # load and parse monopolar channels
    mono_tsv = f'{prefix}_acq-monopolar_channels.tsv'
    mono_json = f'{prefix}_acq-monopolar_ieeg.json'
    mono_df = pd.read_csv(mono_tsv, sep='\t')
    mono_dict = json.load(open(mono_json))

    mono_nonzero_counts = [key for key in mono_dict if 'ChannelCount' in key and mono_dict[key] > 0]
    mono_channels = {key.replace('ChannelCount', ''): mono_dict[key] for key in mono_nonzero_counts}
    groups = np.unique(mono_df['group']).tolist()
    record_dur = mono_dict['RecordingDuration']
    freq = mono_dict['SamplingFrequency']

    # if bipolar channels exist, load and compare with monopolar channels
    bi = True
    bi_tsv = f'{prefix}_acq-bipolar_channels.tsv'
    bi_json = f'{prefix}_acq-bipolar_ieeg.json'
    try:
        bi_df = pd.read_csv(bi_tsv, sep='\t')
        bi_dict = json.load(open(bi_json))
    except FileNotFoundError:
        bi = False

    if bi:
        bi_nonzero_counts = [key for key in bi_dict if 'ChannelCount' in key and bi_dict[key] > 0]
        bi_channels = {key.replace('ChannelCount', ''): bi_dict[key] for key in bi_nonzero_counts}
        assert len(groups) == len(np.unique(bi_df['group'])), 'number of unique channel groups do not match'
        assert record_dur == bi_dict['RecordingDuration'], 'recording durations do not match'
        assert freq == bi_dict['SamplingFrequency'], 'sampling frequencies do not match'
    else:
        bi_channels = {}

    # print information
    if print_info:
        print(f'Info for subject {sub} session {ses}')
        print('-'*60)
        print(f'monopolar channel types + counts: {mono_channels}')
        if bi:
            print(f'bipolar channel types + counts: {bi_channels}')
        else:
            print('bipolar channels: MISSING')
        print(f'number of unique channel groups: {len(groups)}')
        print(f'recording time: {round(record_dur/3600, 4)} hrs')
        print(f'sampling frequency: {freq} Hz')

    return mono_channels, bi_channels, groups, record_dur, freq

def get_ses_agg_info(path, subjects, max_ses, print_info=True):
    '''
    return and print session-aggregated data for each given subject

    args:
        path (str): path to the task folder (not the subject folder)
        subjects (list): list of subject IDs to parse
        print_info (bool): whether to print the info or not
    
    returns:
        sub_info (dict): dictionary of subject IDs and their session-aggregated data
    '''
    tasknum = path.split('/')[-1]
    agg_info = {'tasknum': tasknum} # across all sessions

    for sub in subjects:
        sub_dict = {
            'sessions': 0,
            'electrodes (mono)': {},
            'record time (hrs)': 0,
            'sample freq (Hz)': []
        }

        ses_count = 0
        for ses in range(max_ses):
            folder = f'{path}/{sub}/ses-{ses}/'
            if not os.path.exists(folder):
                continue
                
            mono, bi, groups, dur, freq = get_sub_info(path, sub, ses, print_info=False)

            assert len(mono) == len(sub_dict['electrodes (mono)']) or len(sub_dict['electrodes (mono)']) == 0, f'Inconsistent electrode types for {sub} ses-{ses}'
            sub_dict['electrodes (mono)'] = mono

            sub_dict['record time (hrs)'] += dur/3600
            sub_dict['sample freq (Hz)'].append(freq)

            ses_count += 1
        
        if ses_count == 0:
            continue

        sub_dict['sessions'] = ses_count
        sub_dict['record time (hrs)'] = round(sub_dict['record time (hrs)'], 4)
        if len(set(sub_dict['sample freq (Hz)'])) == 1:
            sub_dict['sample freq (Hz)'] = sub_dict['sample freq (Hz)'][0]

        agg_info[sub] = sub_dict

    if print_info:
        print(f'Session-aggregated info for {len(agg_info)-1} subjects for task {tasknum}')
        print(f'{"-"*60}')
        for sub in agg_info:
            if sub == 'tasknum':
                continue
            print(f'{tasknum} Subject {sub}')
            print(agg_info[sub])
            print()
    
    return agg_info

def get_sub_ses_agg_info(ses_agg_info, print_info=True):
    tasknum = ses_agg_info['tasknum']
    subs_count = len(ses_agg_info)

    sessions = 0
    electrodes = {}
    record_dur = 0

    for sub in ses_agg_info:
        if sub == 'tasknum':
            continue
        sessions += ses_agg_info[sub]['sessions']
        record_dur += ses_agg_info[sub]['record time (hrs)']

        for electrode in ses_agg_info[sub]['electrodes (mono)']:
            count = ses_agg_info[sub]['electrodes (mono)'][electrode]
            if electrode not in electrodes:
                electrodes[electrode] = count
            else:
                electrodes[electrode] += count
    
    if print_info:
        print(f'Subject and session info aggregated over {subs_count} subjects for task {tasknum}:')
        print('-'*60)    
        print(f'Number of sessions: {sessions}, avg {sessions/subs_count:.2f} sessions per subject')
        elec_sub_avg = {elec:round(count/subs_count,4) for elec,count in electrodes.items()}
        print(f'Number of electrodes: {electrodes}, avg {elec_sub_avg} per subject (unchanging for each subject)')
        print(f'Recording time: {record_dur:.2f} hrs, avg {record_dur/subs_count:.2f} hrs per subject, {record_dur/sessions:.2f} hrs per session')

    return subs_count, sessions, electrodes, record_dur
