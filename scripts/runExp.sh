#!/bin/bash
## Run some experiments

# Train 
for seed in 0; do
#    ## Default is 20Hz control frequency, 1 action history, 3s episode length, 200 limit for jerk action
    #python scripts/train.py --task=NPThrow3 --num_envs 4096 --experiment_name Throw_1hist_20Hz_200Limit_02muRobust --seed $seed --headless
    #python scripts/train.py --task=NPThrow3 --num_envs 4096 --experiment_name Throw_1hist_20Hz_200Limit_6DOF --seed $seed --headless
    #python scripts/train.py --task=NPThrow3 --num_envs 4096 --experiment_name Throw_1hist_20Hz_200Limit_underOver --seed $seed --headless

#    for ctrlFreq in 10 30 60; do
#        python scripts/train.py --task=NPThrow3 --num_envs 4096 --experiment_name Throw_1hist_${ctrlFreq}Hz_200Limit --seed $seed --headless --ctrlFreq $ctrlFreq
#    done
#
#    for histLen in 3 5 10; do
#        python scripts/train.py --task=NPThrow3 --num_envs 4096 --experiment_name Throw_${histLen}hist_20Hz_200Limit --seed $seed --headless --histLen $histLen
#    done
#
#    for jerkLimit in 50 100 400; do
#        python scripts/train.py --task=NPThrow3 --num_envs 4096 --experiment_name Throw_1hist_20Hz_${jerkLimit}Limit --seed $seed --headless --jerkLimit $jerkLimit
#    done

    #for DOF in 2; do
    #    python scripts/train.py --task=NPThrow3 --num_envs 4096 --experiment_name Throw_1hist_20Hz_200Limit_${DOF}DOF --seed $seed --headless --DOF $DOF
    #done

    #for controlMode in "vel"; do
    #    python scripts/train.py --task=NPThrow3 --num_envs 4096 --experiment_name Throw_1hist_20Hz_200Limit_${controlMode}Control --seed $seed --headless --controlMode $controlMode
    #done
    for jerkLimit in 150; do
        for noiseLevel in 3 2; do
            python scripts/train.py --task=NPThrow3 --num_envs 4096 --experiment_name Throw_1hist_20Hz_${jerkLimit}Limit_${noiseLevel}muRobust --seed 0 --headless --noiseLevel $(echo "$noiseLevel*0.1" | bc) --jerkLimit $jerkLimit
        done
    done
done

# Eval 
#for seed in 0 1 2; do
    ## Default is 20Hz control frequency, 1 action history, 3s episode length, 200 limit for jerk action
    #python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_200Limit  --seed $seed --targetObject General --envSeed 0
#    python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_200Limit_6DOF  --seed $seed --targetObject General --envSeed 0
    #python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_200Limit_015muRobust  --seed $seed --targetObject General --envSeed 0
    #for ctrlFreq in 10 30 60; do
    #    python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_${ctrlFreq}Hz_200Limit  --seed $seed --targetObject General --envSeed 0 --ctrlFreq $ctrlFreq
    #done
#
    #for histLen in 3 5 10; do
    #    python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_${histLen}hist_20Hz_200Limit --seed $seed --targetObject General --envSeed 0 --histLen $histLen
    #done
#
    #for jerkLimit in 50 100 400; do
    #    python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_${jerkLimit}Limit --seed $seed --targetObject General --envSeed 0 --jerkLimit $jerkLimit
    #done

    #for DOF in 2; do
    #    python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_200Limit_${DOF}DOF  --seed $seed --targetObject General --envSeed 0 --DOF $DOF
    #done

    #for controlMode in "vel" "acc"; do
    #    python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_200Limit_${controlMode}Control  --seed $seed --targetObject General --envSeed 0 --controlMode $controlMode
    #done

    #python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_100Limit_Opt --targetObject uncertainMu_00 --seed 0 --envSeed $seed
    #python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_100Limit_01muRobust --targetObject uncertainMu_00 --seed 0 --envSeed $seed
    #python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_100Limit_015muRobust --targetObject uncertainMu_00 --seed 0 --envSeed $seed
    #python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_100Limit_02muRobust --targetObject uncertainMu_00 --seed 0 --envSeed $seed
    #python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_100Limit_025muRobust --targetObject uncertainMu_00 --seed 0 --envSeed $seed
    #python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_100Limit_fullmuRobust --targetObject uncertainMu_025 --seed 0 --envSeed $seed

#done

#Eval uncertainy
#for seed in 0 1 2; do
#    for noiseLevel in 0 1 2 3 4 5; do
        #for noiseLevelModel in 1 2 3 4 5; do
        #    python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_100Limit_${noiseLevelModel}muRobust --seed 0 --targetObject uncertainMu_${noiseLevel} --envSeed $seed --noiseLevel $(echo "$noiseLevel*0.1" | bc)
        #done
        #python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_100Limit_Opt --seed 0 --targetObject uncertainMu_${noiseLevel} --envSeed $seed --noiseLevel $(echo "$noiseLevel*0.1" | bc)
#        python scripts/Eval.py --task=NPThrow3 --num_envs 4096 --headless --experiment_name Throw_1hist_20Hz_100Limit_fullmuRobust --seed 0 --targetObject uncertainMu_${noiseLevel} --envSeed $seed --noiseLevel $(echo "$noiseLevel*0.1" | bc)

#    done
#done


