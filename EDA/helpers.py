import pandas as pd
import numpy as np
import json
import os

def get_electrodes(mono_df, bi_df):
    '''
    Return a dictionary mapping electrode types to their unique electrode names,
    combining data from monopolar and bipolar channel DataFrames.

    Args:
        mono_df (pd.DataFrame): DataFrame containing monopolar channels.
                                Expected to have columns 'type' and 'name'.
        bi_df (pd.DataFrame): DataFrame containing bipolar channels.
                              Expected to have columns 'type' and 'name',
                              where the 'name' column contains bipolar pair strings (e.g., "A1-A2").

    Returns:
        electrodes (dict): Dictionary where each key is an electrode type (from either DataFrame)
                           and the value is a list of unique electrode names for that type.
    '''
    mono_electrodes = {}
    bi_electrodes = {}

    if mono_df is not None and not mono_df.empty:
        mono_electrodes = {
            t: mono_df[mono_df['type'] == t]['name'].tolist()
            for t in mono_df['type'].unique()
        }

    if bi_df is not None and not bi_df.empty:
        for t in bi_df['type'].unique():
            pairs = bi_df[bi_df['type'] == t]['name'].tolist()
            electrodes = []
            for pair in pairs:
                electrodes += pair.split('-')
            bi_electrodes[t] = list(set(electrodes))
    
    # Combine the electrode lists for all types (union of keys) using .get() to default to empty list if key is missing.
    all_electrodes = {
        t: list(set(mono_electrodes.get(t, []) + bi_electrodes.get(t, [])))
        for t in set(mono_electrodes.keys()) | set(bi_electrodes.keys())
    }
    
    return all_electrodes

def get_sub_info(ds_num, tasknum, sub, ses, print_info=True, log_path=None):
    '''
    return and print (mono and bi) channels + channel groups, recording time, and sampling frequency info for a given subject and session
    subject sessions without ieeg data will be ignored

    args:
        ds_num (str): dataset id
        tasknum (str): task + number
        sub (str): subject ID
        ses (int): session number
        print_info (bool): whether to print the info or not
    
    returns:
        mono_channels (dict): dictionary of monopolar channel types and counts
        bi_channels (dict): dictionary of bipolar channel types and counts
        electrodes (dict): dictionary of electrode types and their names
        groups (list): list of unique channel groups
        record_dur (float): recording duration in SECONDS
        freq (int): sampling frequency in Hz
    '''
    # get file paths
    ieeg_prefix = f'{ds_num}/{sub}/ses-{ses}/ieeg'
    if not os.path.exists(ieeg_prefix): # check if ieeg folder exists
        if print_info:
            print(f'No ieeg folder for {tasknum} subject {sub} session {ses}')
        if log_path != None:
            with open(log_path, 'a') as f:
                f.write(f'No ieeg folder for {tasknum} subject {sub} session {ses}\n\n')
        return None, None, None, None, None, None
    prefix = ieeg_prefix + f'/{sub}_ses-{ses}_task-{tasknum}'

    # if they exist, parse and load monopolar channels
    mono = True
    mono_tsv = f'{prefix}_acq-monopolar_channels.tsv'
    mono_json = f'{prefix}_acq-monopolar_ieeg.json'
    try:
        mono_df = pd.read_csv(mono_tsv, sep='\t')
        mono_dict = json.load(open(mono_json))
    except FileNotFoundError:
        mono = False
        mono_df = pd.DataFrame([]) # empty df

    if mono:
        mono_nonzero_counts = [key for key in mono_dict if 'ChannelCount' in key and mono_dict[key] > 0]
        mono_channels = {key.replace('ChannelCount', ''): mono_dict[key] for key in mono_nonzero_counts}
        groups = np.unique(mono_df['group']).tolist()
        record_dur = mono_dict['RecordingDuration']
        freq = mono_dict['SamplingFrequency']
    else:
        mono_channels = {}
        groups = []
        record_dur = 0
        freq = 0

    # if bipolar channels exist, parse and load
    bi = True
    bi_tsv = f'{prefix}_acq-bipolar_channels.tsv'
    bi_json = f'{prefix}_acq-bipolar_ieeg.json'
    try:
        bi_df = pd.read_csv(bi_tsv, sep='\t')
        bi_dict = json.load(open(bi_json))
    except FileNotFoundError:
        bi = False
        bi_df = pd.DataFrame([]) # empty df

    if bi:
        bi_nonzero_counts = [key for key in bi_dict if 'ChannelCount' in key and bi_dict[key] > 0]
        bi_channels = {key.replace('ChannelCount', ''): bi_dict[key] for key in bi_nonzero_counts}
        if not mono:
            groups = np.unique(bi_df['group']).tolist()
            record_dur = bi_dict['RecordingDuration']
            freq = bi_dict['SamplingFrequency']
    else:
        bi_channels = {}
    
    # if no channels exist, return Nones
    if not mono and not bi:
        if print_info:
            print(f'No channel data for {tasknum} subject {sub} session {ses}')
        if log_path != None:
            with open(log_path, 'a') as f:
                f.write(f'No channel data for {tasknum} subject {sub} session {ses}\n\n')
        return None, None, None, None, None, None

    # get electrodes
    electrodes = get_electrodes(mono_df, bi_df)
    agg_counts = {t: len(electrodes[t]) for t in electrodes.keys()}
    if len(electrodes) == 0:
        print('something is very very wrong - no electrodes after checking mono and bi are nonempty')
        print(sub, ses)
        return None

    # information for printing and logging
    info = [
        f'Info for subject {sub} session {ses}',
        '-'*60,
        f'monopolar channel types + counts: {mono_channels}',
        f'bipolar channel types + counts: {bi_channels}' if bi else 'bipolar channels: MISSING',
        f'aggregated electrode counts: {agg_counts}',
        f'number of unique channel groups: {len(groups)}',
        f'recording time: {round(record_dur/3600, 4)} hrs',
        f'sampling frequency: {freq} Hz'
    ]

    if print_info:
        for line in info:
            print(line)

    if log_path != None:
        with open(log_path, 'a') as f:
            for line in info:
                f.write(line + '\n')
            f.write('\n')

    return mono_channels, bi_channels, electrodes, groups, record_dur, freq

def get_ses_agg_info(ds_num, tasknum, subjects, max_ses, print_info=True, log_path=None):
    '''
    return and print session-aggregated data for each given subject
    subject sessions without ieeg data will be ignored

    args:
        ds_num (str): dataset id
        tasknum (str): task + number
        subjects (list): list of subject IDs to parse
        print_info (bool): whether to print the info or not
    
    returns:
        sub_info (dict): dictionary of subject IDs and their session-aggregated data
    '''
    agg_info = {} # across all sessions

    for sub in subjects:
        sub_dict = {
            'sessions': 0,
            'electrodes': {},
            'record time (hrs)': 0,
            'sample freq (Hz)': []
        }

        ses_count = 0
        for ses in range(max_ses):
            folder = f'{ds_num}/{sub}/ses-{ses}/'
            if not os.path.exists(folder):
                continue
                
            mono, bi, electrodes, groups, dur, freq = get_sub_info(ds_num, tasknum, sub, ses, print_info=False)
            if not electrodes and not mono and not bi:
                continue
            electrodes_counts = {t: len(electrodes[t]) for t in electrodes.keys()}

            # inconsistent counts of electrodes in same subject -> faulty electrodes -> choose largest counts
            sub_dict['electrodes'] = max(sub_dict['electrodes'], electrodes_counts, key=lambda x: sum(list(x.values())))

            sub_dict['record time (hrs)'] += dur/3600
            sub_dict['sample freq (Hz)'].append(freq)
            
            ses_count += 1
        
        if ses_count == 0:
            continue

        sub_dict['sessions'] = ses_count
        sub_dict['record time (hrs)'] = round(sub_dict['record time (hrs)'], 4)
        sub_dict['sample freq (Hz)'] = list(set(sub_dict['sample freq (Hz)']))
        if len(sub_dict['sample freq (Hz)']) == 1:
            sub_dict['sample freq (Hz)'] = sub_dict['sample freq (Hz)'][0]

        agg_info[sub] = sub_dict
    
    # information for printing and logging
    info = [
        f'Session-aggregated info for {len(agg_info)} valid subjects for task {tasknum}',
        '-'*60
    ]
    info += [f'{tasknum} Subject {sub}\n{agg_info[sub]}\n' for sub in agg_info if sub != 'tasknum']

    if print_info:
        for line in info:
            print(line)
    
    if log_path != None:
        with open(log_path, 'a') as f:
            for line in info:
                f.write(line + '\n')
            f.write('\n')
    
    return agg_info

def get_sub_ses_agg_info(ses_agg_info, tasknum, og_all_subs, print_info=True, log_path=None):
    '''
    return and print session-aggregated data for each given subject

    args:
        ses_agg_info (dict): session-aggregated data for each subject
        tasknum (str): task + number
        print_info (bool): whether to print the info or not

    returns:
        subs_count (int): number of subjects
        sessions (int): total number of sessions
        electrodes (dict): dictionary of electrode types and counts
        record_dur (float): total recording duration in hours
    '''
    subs_count = len(ses_agg_info)

    sessions = 0
    electrodes = {}
    record_dur = 0
    freqs = {}

    for sub in ses_agg_info:
        sessions += ses_agg_info[sub]['sessions']
        record_dur += ses_agg_info[sub]['record time (hrs)']

        if type(ses_agg_info[sub]['sample freq (Hz)']) == list:
            freq_info = [freq for freq in ses_agg_info[sub]['sample freq (Hz)']]
        else:
            freq_info = [ses_agg_info[sub]['sample freq (Hz)']]
        for freq in freq_info:
            if freq not in freqs:
                freqs[freq] = 1
            else:
                freqs[freq] += 1

        for electrode in ses_agg_info[sub]['electrodes']:
            count = ses_agg_info[sub]['electrodes'][electrode]
            if electrode not in electrodes:
                electrodes[electrode] = count
            else:
                electrodes[electrode] += count
    
    # information for printing and logging
    elec_sub_avg = {elec:round(count/subs_count,4) for elec,count in electrodes.items()}
    info = [
        f'Subject and session info aggregated over {subs_count} valid subjects for task {tasknum}',
        '-'*60,
        f'Number of sessions: {sessions}, avg {sessions/subs_count:.2f} sessions per subject',
        f'Number of electrodes: {electrodes}, avg {elec_sub_avg} per subject (unchanging for each subject)',
        f'Recording time: {record_dur:.2f} hrs, avg {record_dur/subs_count:.2f} hrs per subject, {record_dur/sessions:.2f} hrs per session',
        f'Sampling frequencies (Hz): {freqs}',
        f'Invalid subject(s): {list(set(og_all_subs) - set(ses_agg_info.keys()))}',
    ]

    if print_info:
        for line in info:
            print(line)

    if log_path != None:
        with open(log_path, 'a') as f:
            for line in info:
                f.write(line + '\n')
            f.write('\n')

    return subs_count, sessions, electrodes, record_dur
