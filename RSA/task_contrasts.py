# Chosen contrasts for modeling the FPN

tasks_contrasts = {
    'ArchiStandard': [
        'video_computation', # mental subtraction upon video instruction
        'audio_computation', # mental subtraction upon audio instruction
        'computation', # mental subtraction (audio_computation & video_computation?)
        'computation-sentences' # mental subtraction vs. sentence reading (audio_computation & video_computation?)
    ],

    'Attention': [
        'spatial_cue - double_cue', # cued vs. uncued probe (spatial attention?)
        'spatial_incongruent - spatial_congruent', # ignore distractors vs. no distractors with spatial cue (attentional control?)
        'double_incongruent - double_congruent', # ignore distractors vs. no distractors without spatial cue (cognitive control?)
        'incongruent - congruent' # ignore distractors vs. no distractors
    ],

    'Catell': [
        'easy', # easy oddball task
        'hard', # hard oddball task
        'hard-easy' # easy vs hard oddball task
    ],

    'ColumbiaCards': [
        'gain', # expected gain in gambling
        'loss', # expected loss in gambling
        'num_loss_cards' # probability of losing in gambling
    ],

    'Discount': [
        'amount', # effect of reward gain
        'delay' # effect of delay on reward
    ],

    'Enumeration': [
        'enumeration_constant', # enumeration
        'enumeration_linear', # linear effect of numerosity in enumeration
        'enumeration_quadratic' # quadratic effect of numerosity in enumeration (interaction)
    ],

    'HcpGambling': [
        'punishment', # negative gambling outcome
        'punishment-reward', # negative vs. positive gambling outcome
        'reward', # gambling with positive outcome
        'reward-punishment' # positive vs. negative gambling outcome
    ],

    'HcpRelational': [
        'relational-match', # relational comparison vs. fixation
        'relational' # relational comparison vs. matching
    ],

    'HcpWm': 
    '',

    'MVEB': 
    '',

    'MathLanguage': 
    '',

    'SelectiveStopSignal': 
    '',

    'StopNogo': 
    '',

    'StopSignal': 
    '',

    'Stroop': 
    '',

    'TwoByTwo': 
    '',

    'VSTM': 
    '',

    'VSTMC': 
    '',

    'WardAndAllport': 
    ''
}