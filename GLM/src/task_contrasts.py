# SOLVE QUESTION: USE CONTRASTS OR INDIVIDUAL CONDITIONS TO BUILD BETA MAPS?

tasks_contrasts = {
    'ArchiStandard': [
        'audio_computation',              # Mental subtraction upon audio instruction
        'audio_left_button_press',       # Left hand button presses upon audio instructions
        'audio_right_button_press',      # Right hand button presses upon audio instructions
        'audio_sentence',                # Listen to narrative sentence
        'cognitive-motor',                # Contrast: Narrative/computation conditions vs. button presses
        'computation',                    # Mental subtraction (pure computation condition)
        'computation-sentences',          # Contrast: Mental subtraction vs. sentence reading
        'horizontal-vertical',           # Contrast: Horizontal vs. vertical checkerboard
        'horizontal_checkerboard',       # Watch horizontal checkerboard
        'left-right_button_press',       # Contrast: Left vs. right hand button press
        'listening-reading',             # Contrast: Listening to a sentence vs. reading a sentence
        'motor-cognitive',               # Contrast: Button presses vs. narrative/computation conditions
        'reading-checkerboard',          # Contrast: Read sentence vs. checkerboard
        'reading-listening',             # Contrast: Reading sentence vs. listening to sentence
        'right-left_button_press',       # Contrast: Right vs. left hand button press
        'sentences',                     # Read or listen to sentences (general language processing)
        'sentences-computation',         # Contrast: Sentence reading vs. mental subtraction
        'vertical-horizontal',           # Contrast: Vertical vs. horizontal checkerboard
        'vertical_checkerboard',         # Watch vertical checkerboard
        'video_computation',             # Mental subtraction upon video instruction
        'video_left_button_press',       # Left hand button presses upon video instructions
        'video_right_button_press',      # Right hand button presses upon video instructions
        'video_sentence'                 # Read narrative sentence presented via video
    ],

    'Attention': [
        'double_congruent',                      # No spatial cue and no distractors in the probe
        'double_cue',                            # Cues appear simultaneously in both possible probe locations
        'double_incongruent',                     # No spatial cue with distractors in the probe
        'double_incongruent-double_congruent',    # Contrast: ignore distractors vs. no distractors without spatial cue
        'incongruent-congruent',                  # Contrast: ignore distractors vs. no distractors (general conflict effect)
        'spatial_congruent',                     # Cued probe with no distractors
        'spatial_cue',                           # Cued probe condition (general cueing effect)
        'spatial_cue-double_cue',                 # Contrast: cued vs. uncued probe, isolating the effect of spatial cueing
        'spatial_incongruent',                   # Cued probe with distractors in the probe
        'spatial_incongruent-spatial_congruent'   # Contrast: ignore distractors vs. no distractors with spatial cue
    ],

    'Catell': [
        'easy',     # easy oddball task
        'hard',     # hard oddball task
        'hard-easy' # easy vs hard oddball task
    ],

    'ColumbiaCards': [
        'gain',          # expected gain in gambling
        'loss',          # expected loss in gambling
        'num_loss_cards' # probability of losing in gambling
    ],

    'Discount': [
        'amount',   # effect of reward gain
        'delay'     # effect of delay on reward
    ],

    'Enumeration': [
        'enumeration_constant', # enumeration
        'enumeration_linear',   # linear effect of numerosity in enumeration
        'enumeration_quadratic' # quadratic effect of numerosity in enumeration (interaction)
    ],

    'HcpGambling': [
        'punishment',        # negative gambling outcome
        'punishment-reward', # negative vs. positive gambling outcome
        'reward',            # gambling with positive outcome
        'reward-punishment'  # positive vs. negative gambling outcome
    ],

    'HcpRelational': [
        'match',            # relational matching
        'relational',       # relational comparison vs. matching
        'relational-match'  # relational comparison vs. fixation
    ],

    'HcpWm': [
        '0back-2back',  # 0-back vs. 2-back task
        '0back_body',   # Body image 0-back task vs. fixation
        '0back_face',   # Face image 0-back task vs. fixation
        '0back_place',  # Place image 0-back task vs. fixation
        '0back_tools',  # Tool image 0-back task vs. fixation
        '2back-0back',  # 2-back vs. 0-back contrast capturing working memory load and frontoparietal engagement
        '2back_body',   # Body image 2-back task vs. fixation
        '2back_face',   # Face image 2-back task vs. fixation
        '2back_place',  # Place image 2-back task vs. fixation
        '2back_tools',   # Tool image 2-back task vs. fixation
        'body-avg',     # Body image versus face place tool image
        'face-avg',     # Face image versus body place tool image
        'place-avg',    # Place image versus face body tool image
        'tools-avg',    # Tool image versus face place body image
    ],

    'MVEB': [
        '2_letters_different',                     # maintaining two letters
        '2_letters_different-same',                # maintaining two letters vs. one
        '2_letters_same',                          # maintaining one letter
        '4_letters_different',                     # maintaining four letters
        '4_letters_different-same',                # maintaining four letters vs. one
        '4_letters_same',                          # maintaining one letter
        '6_letters_different',                     # maintaining six letters
        '6_letters_different-2_letters_different', # maintaining six letters vs. two
        '6_letters_different-same',                # maintaining six letters vs. one
        '6_letters_same',                          # maintaining one letter
        'letter_occurrence_response',              # respond by button pressing whether the letter currently displayed was presented before or not
    ],

    'MathLanguage': [
        'arithmetic_fact-othermath',           # arithmetic fact vs other maths
        'arithmetic_fact_auditory',            # listen to arithmetic fact
        'arithmetic_fact_visual',              # read arithmetic fact
        'arithmetic_principle-othermath',      # arithmetic principle vs other maths
        'arithmetic_principle_auditory',       # listen to arithmetic principle
        'arithmetic_principle_visual',         # read to arithmetic principle
        'auditory-visual',                    # listen to vs read instruction
        'colorlessg-wordlist',                 # jabberwocky vs word list
        'colorlessg_auditory',                 # auditory jabberwocky sentence parsing
        'colorlessg_visual',                   # visual jabberwocky sentence parsing
        'context-general',                    # cue vs language statement
        'context-theory_of_mind',             # cue vs false belief
        'context_auditory',                   # audio cue
        'context_visual',                     # visual cue
        'general-colorlessg',                 # listen to sentence vs jabberwocky
        'general_auditory',                   # listen to sentence
        'general_visual',                     # read sentence
        'geometry-othermath',                  # geometry vs other maths
        'geometry_fact_auditory',              # listen to geometric fact
        'geometry_fact_visual',                # read geometric fact
        'math-nonmath',                        # math vs others
        'nonmath-math',                       # others vs math
        'theory_of_mind-context',              # false belief vs cue
        'theory_of_mind-general',              # false belief vs general statement
        'theory_of_mind_and_context-general',  # false belief and cue vs general statement
        'theory_of_mind_auditory',             # auditory false-belief tale
        'theory_of_mind_visual',               # read false-belief tale
        'visual-auditory',                    # read vs to listen to instruction
        'wordlist_auditory',                  # listen to word list
        'wordlist_visual'                     # read word list
    ],

    'SelectiveStopSignal': [
        'go_critical',           # Respond with the correct finger (side instructed to stop if a stop signal appears)
        'go_critical-stop',      # Contrast: Inhibit motor response vs. executing go_critical response
        'go_noncritical',        # Respond with the correct finger (side instructed to ignore the stop signal)
        'go_noncritical-ignore', # Contrast: Ignoring the stop signal vs. simply responding in go_noncritical
        'ignore',                # Respond anyway even if the stop signal appears
        'ignore-stop',           # Contrast: Comparing ignore condition vs. successfully inhibiting (stop)
        'stop',                  # Successfully stop the response when the stop signal appears
        'stop-ignore'            # Contrast: Inhibit motor response (stop) vs. ignoring the stop signal
    ],

    'StopNogo': [
        'go',                          # Shape recognition baseline: response executed (go condition)
        'nogo',                        # No response condition (nogo), isolating inhibitory processing
        'nogo-go',                     # Contrast capturing response inhibition (inhibiting a prepotent response)
        'successful+nogo-unsuccessful',# Contrast indexing failed inhibition by comparing successful no-go vs. unsuccessful inhibition
        'successful_stop',             # Shape recognition when the response is successfully inhibited (stopped response)
        'unsuccessful-successful_stop',# Contrast highlighting the effect of failed inhibition (unsuccessful vs. successful stopping)
        'unsuccessful_stop',           # Shape recognition when the stopping attempt fails (unsuccessful inhibition)
    ],

    'StopSignal': [
        'go',      # Shape recognition baseline: response executed during go condition
        'stop',    # Shape recognition with successful stopping (inhibited response)
        'stop-go'  # Contrast capturing response inhibition by comparing stop vs. go condition
    ],

    'Stroop': [
        'congruent',                # Word and word color are the same (baseline condition)
        'incongruent',              # Word and color are not the same, inducing interference
        'incongruent-congruent'     # Contrast isolating conflict: automatic reading vs. instructed response
    ],

    'TwoByTwo': [
        'cue_switch-stay',             # Isolates the effect of switching the cue versus staying (cue-level effect)
        'cue_taskstay_cuestay',       # Baseline condition where both the task and cue repeat (no switch)
        'cue_taskswitch_cuestay',      # Condition where the task switches while the cue remains the same (isolates task switch at the cue level)
        'cue_taskswitch_cueswitch',    # Condition where both the task and cue switch (combined switching effect at the cue level)
        'stim_taskstay_cuestay',      # Baseline condition at the stimulus level where both task and cue repeat
        'stim_taskstay_cueswitch',     # Condition at the stimulus level with task repeating but cue switching (isolates cue switch effect in stimulus processing)
        'stim_taskswitch_cuestay',     # Condition at the stimulus level with task switching while the cue remains the same (isolates task switch effect in stimulus processing)
        'stim_taskswitch_cueswitch',   # Condition at the stimulus level where both task and cue switch (combined switching effect in stimulus processing)
        'task_switch-stay'             # Isolates the overall effect of switching the task versus staying (task-level effect)
    ],

    'VSTM': [
        'vstm_constant',   # Visual orientation baseline condition in visual short-term memory
        'vstm_linear',     # Captures the linear effect of numerosity in visual orientation (gradual increase with more items)
        'vstm_quadratic'   # Captures the quadratic effect of numerosity in visual orientation (non-linear relationship, e.g., saturation or threshold effects)
    ],

    'VSTMC': [
        'resp',               # Baseline: general response to motion
        'resp_load1',         # Response to motion direction when attending to one set of points (low load)
        'resp_load2',         # Response to motion direction when attending to two sets of points (moderate load)
        'resp_load3',         # Response to motion direction when attending to three sets of points (high load)
        'resp_load3-load1',   # Contrast: Difference in response between high load (three sets) and low load (one set)
        'stim',               # Baseline: attending to sets of points
        'stim_load1',         # Attending to one set of points (low attentional load)
        'stim_load2',         # Attending to two sets of points (moderate attentional load)
        'stim_load3',         # Attending to three sets of points (high attentional load)
        'stim_load3-load1'    # Contrast: Difference in attention between high load (three sets) and low load (one set)
    ],

    'WardAndAllport': [
        'ambiguous-unambiguous',         # Effect of goal hierarchy: complex (ambiguous) vs. simple (unambiguous)
        'intermediate-direct',           # Effect of search depth: complex (intermediate) vs. simple (direct)
        'move_ambiguous_direct',         # Movement phase: complex goal hierarchy + simple search depth
        'move_ambiguous_intermediate',   # Movement phase: complex goal hierarchy + complex search depth
        'move_unambiguous_direct',       # Movement phase: simple goal hierarchy + simple search depth
        'move_unambiguous_intermediate', # Movement phase: simple goal hierarchy + complex search depth
        'planning_ambiguous_direct',     # Planning phase: complex goal hierarchy + simple search depth
        'planning_ambiguous_intermediate',# Planning phase: complex goal hierarchy + complex search depth
        'planning_unambiguous_direct',    # Planning phase: simple goal hierarchy + simple search depth
        'planning_unambiguous_intermediate' # Planning phase: simple goal hierarchy + complex search depth
    ],
}