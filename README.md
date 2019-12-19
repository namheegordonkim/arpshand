# A Rapid Prototyping System for Hand Gesture Recognition
Nam Hee Gordon Kim and Tim Straubinger

This is a supplementary code for our project: https://sites.google.com/view/arpshand

We used Python 3.6 on Ubuntu 18.04 LTS along with these libraries (not comprehensive):
* matplotlib
* scikit-learn
* imageio
* tensorflow

Also, an installation of Docker or Singularity is recommended.

## Instructions

### Step 1: Deploy the 3D hand pose estimator (Zimmermann et al.)

* We employed a remote server with a GPU and used socket programming to process videos.
* We provide the Dockerfile used inside the hand3d subdirectory. If running Docker, `docker pull namheegordonkim/handgpu` should suffice.
* Use `docker shell` or similar to access the files inside the container.
* Once inside the container, run `python3 -u server.py` to listen to port 3333.
* If needed, set up an SSH tunnel so the client running on your edge device can communicate with the remote server.

### Step 2: Preprocess data

You can download our video data here: https://www.dropbox.com/s/ep8jhys2ie4kjda/smash-g-data.zip?dl=0

Put all the .mp4 files inside a subdirectory named `./data/`.

To process all the data in one command, run this inside a bash shell:

```
for GESTURE in "ok" "thumbs_up" "paper" "scissors" "call_me" "lets_drink"
do
    for i in {00..08}
    do
        python preprocess_video_with_cc.py --input_file ./data/"$GESTURE"$i.mp4 --output_dir ./data/dynamic/$GESTURE/$i/
    done
done
```

You should be able to visualize the results of the data by using e.g.

```
python visualize_angles.py --angles_dir ./data/dynamic/paper/00/angles
```

### Step 3: Learn the models

Run these to generate the needed pickle files:

```
python learn_clutch.py
python learn_clutch_corrector.py
python learn_onedollar.py
```

### Step 4: Run the live predictor

While the 3D hand pose estimator is deployed, run

```
python live_predictor.py
```

