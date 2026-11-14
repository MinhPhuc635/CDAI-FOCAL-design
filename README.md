# Vehicle Acoustic–Seismic Classifier

A Streamlit application that classifies vehicles from synchronized audio and seismic signals using the FOCAL model. It displays the predicted class, confidence score, and probabilities for other likely classes.

## Open the app

Use the deployed application:

**[https://cdai-focal-1.onrender.com/](https://cdai-focal-1.onrender.com/)**

You do not need to install Python or train the model to use the deployed app. The server must include the FOCAL checkpoint and its label and metrics files. The service may take a short time to start when opened.

## Dataset and input files

The dataset is available on Kaggle:

**[FOCAL Dataset](https://www.kaggle.com/datasets/nguyenmp/focal-dataset)**

Each recording needs a synchronized audio signal and vertical EHZ seismic signal:

- **Audio:** a `.wav` file or a MOD-format CSV whose name usually starts with `aud` or `audio`, such as `aud.csv` or `aud16000.csv`.
- **Seismic:** an EHZ CSV, usually named `ehz.csv`.

The app accepts `.csv` and `.wav` files. Keep the audio and EHZ files from the same recording together. Uploading only one signal is not enough to make a prediction.

The Kaggle dataset is for downloading and research. The deployed Render app can only read files uploaded through the browser; it cannot access the `F:` drive on your computer.

## Use the app

1. Open the [Vehicle Acoustic–Seismic Classifier](https://cdai-focal-1.onrender.com/).
2. In **Input Data**, upload the audio file (`.wav` or `aud*.csv`) and the matching EHZ seismic CSV. You can select multiple files at once.
3. Check the uploaded file list. If the app does not identify a signal automatically, select the correct audio and EHZ files manually.
4. Select **Run Prediction**.
5. Read the result in **Results**: the predicted vehicle class, prediction confidence, and probabilities for other classes. If the signal is weak, invalid, or ambiguous, the app may report **Unknown / possibly outside the dataset** instead of forcing a class label.

FOCAL splits the signals into two-second windows for inference. The current Render upload limits are **16 MB per file** and **24 MB total per upload**.

## Understand the results

- **Prediction confidence** is the probability score of the top class, aggregated across signal windows. It is not a guarantee that the prediction is correct at that rate.
- **Test accuracy**, when shown, measures model performance on the dataset's held-out test set. It does not measure whether the current uploaded file was classified correctly.
- **Unknown** means the signal may be outside the conditions the model can reliably recognize, or that the top class probabilities are too close. Check that the files match, are synchronized, and have usable signal quality.

## Run the app locally

Install the Python version required by `requirements.txt`. In PowerShell, run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`.

## Run API and security test cases

Run the automated local API/system and safe security checks with Python's built-in test runner:

```powershell
python -m unittest discover -s tests -v
```

The suite has 20 cases. The real-recording AI test can run with a synchronized pair. To also compare the prediction with ground truth, set the expected label from the dataset:

```powershell
$env:FOCAL_TEST_AUDIO_FILE='D:\MOD_vehicle\raw_data\example\aud.csv'
$env:FOCAL_TEST_SEISMIC_FILE='D:\MOD_vehicle\raw_data\example\ehz.csv'
$env:FOCAL_TEST_EXPECTED_LABEL='bicycle'
python -m unittest discover -s tests -v
```

Replace the example paths and class name with the actual recording and label. If the class label is not known, leave `FOCAL_TEST_EXPECTED_LABEL` unset; the test will still run inference but cannot score correctness. Use a matching audio/seismic pair from the same recording. Security checks must be run only against the local copy or a staging service controlled by admin; they do not perform a public-site scan or a high-volume load test.

## REST API

The project also provides a REST API for integrations and API-level testing. It uses the same FOCAL preprocessing and checkpoint as the Streamlit app. It does not train the model when a request arrives; the checkpoint must already be present.

Start the API in a second PowerShell window:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the interactive API page. Check `GET /api/v1/health`; `checkpoint_ready` should be `true`. To predict, send one audio file (`.wav` or acoustic MOD `.csv`) and its matching EHZ seismic `.csv` to `POST /api/v1/predict` as multipart form fields named `audio` and `seismic`.

PowerShell example:

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/predict" `
  -F "audio=@D:\data\aud.csv" `
  -F "seismic=@D:\data\ehz.csv"
```

Each file is limited to 16 MiB and the pair to 24 MiB. The JSON response contains the predicted class (or `null` when rejected as unknown), confidence, top class probabilities, and test-set accuracy metadata. Confidence is not a promise that this particular prediction is correct.

For a container, build the image as usual, then run the API instead of Streamlit:

```powershell
docker run --rm -p 8000:8000 focal-vehicle-classifier:local uvicorn api:app --host 0.0.0.0 --port 8000
```

For a public deployment, configure a long random `FOCAL_API_KEY` environment variable in the hosting service. When configured, clients must send it in the `X-API-Key` header. Keep the key in the host's secret settings, never in source code. A separate Render web service can use this start command (the existing Streamlit service can remain as it is):

```text
uvicorn api:app --host 0.0.0.0 --port $PORT
```

The first health check should return `checkpoint_ready: true`. If it returns `false`, make sure `models/vehicle_focal.pt` and `models/labels.json` are included in the service image before deploying.

To train the model again from a downloaded dataset, set the environment variable to the dataset's `raw_data` directory and run `train_focal.py`. For example:

```powershell
$env:MOD_VEHICLE_DATA_DIR='F:\MOD_vehicle\raw_data'
python train_focal.py
```

Training requires the dataset and suitable compute resources. Users of the deployed app do not need to train the model.

## Run with Docker

The image runs CPU inference and includes the model checkpoint, labels, and metrics. It does not require the training dataset to make predictions. With Docker Desktop running, build and start the app locally:

```powershell
docker build -t focal-vehicle-classifier:local .
docker run --rm -p 8501:8501 focal-vehicle-classifier:local
```

Open `http://localhost:8501`. The dataset, `.venv`, and training logs should not be added to the image.

## CI/CD with GitHub Actions

The **CI - Build FOCAL image** workflow runs automatically on pushes to `main` and on pull requests. It checks Python syntax and builds the Docker image to catch packaging errors. It does **not** publish the image.

To check a run, open the repository on GitHub, select **Actions**, and open the latest **CI - Build FOCAL image** run:

- A green check and `success` mean the syntax check and Docker build passed.
- A red X and `failure` mean a step failed. Open the failed step to read the error, fix it, and push the change again.

The **Publish FOCAL image to Docker Hub** workflow runs only when started manually. Before publishing, review the changed files and confirm the CI workflow succeeded for the commit you intend to release.

### Configure Docker Hub credentials and approval

Do this once before publishing:

1. Create a Docker Hub Personal Access Token (PAT) with **Repo Read & Write** permission. Read-only permission cannot push images. Docker documents that **Repo Read & Write** allows pushing images to repositories your account manages. [Docker access token permissions](https://docs.docker.com/security/access-tokens/reference/)
2. In the GitHub repository, open **Settings → Secrets and variables → Actions**.
3. Add a repository secret named `DOCKERHUB_USERNAME`. Its value is the Docker Hub login username: `nmp`.
4. Add a second secret named `DOCKERHUB_TOKEN`. Its value is the full PAT. Do not add quotes or the word `Bearer`. Never put the token in source code, README files, or chat messages.
5. Open **Settings → Environments** and select or create the environment named `docker-hub-publish`.
6. Under **Deployment protection rules**, enable **Required reviewers** and add the GitHub account that must approve the release. The current environment needs this reviewer rule to enforce a wait for approval.

GitHub pauses a job that uses an environment with required reviewers until an authorized reviewer approves it. If the person starting the workflow must also approve it, leave **Prevent self-review** disabled. [GitHub environment approvals](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)

### Publish after review

1. In GitHub, open **Actions** and select **Publish FOCAL image to Docker Hub**.
2. Select **Run workflow** and use the `main` branch.
3. Enter `nmp635/cdai-focal` as the image name and `latest` as the tag. A tag is the label used to identify an image version.
4. Start the workflow. If required reviewers are configured, GitHub pauses the job for approval.
5. Review the commit and workflow run, then select **Review deployments → Approve and deploy**.
6. After approval, GitHub Actions logs in to Docker Hub, builds the image, and pushes it. You do not need to run `docker push` manually.

The publish run should finish with a green check. The **Build and publish image** step should show `success`. Confirm the `latest` tag on [Docker Hub: nmp635/cdai-focal](https://hub.docker.com/r/nmp635/cdai-focal).

The image has already been published publicly as `nmp635/cdai-focal:latest`. On another computer with Docker Desktop running, download and run it with:

```powershell
docker pull nmp635/cdai-focal:latest
docker run --rm -p 8501:8501 nmp635/cdai-focal:latest
```

Open `http://localhost:8501`. Because the repository is public, users do not need a Docker Hub token to pull the image.

## Troubleshooting

- **`docker` is not recognized:** Install Docker Desktop, then close and reopen PowerShell.
- **`docker version` shows Client but no Server:** Docker Desktop Engine is not running. Open Docker Desktop, select **Troubleshoot → Restart Docker Desktop**, and wait until the Engine is ready.
- **`read-only file system` while pulling:** This is a Docker Desktop storage issue, not an app error. Restart Docker Desktop and check free disk space. Do not select **Clean up data** or **Reset to factory defaults** unless you have backed up anything you need; these options can delete local images and containers. [Docker Desktop troubleshooting](https://docs.docker.com/desktop/troubleshoot-and-support/troubleshoot/)
- **`authentication required` while pulling the public image:** Run `docker logout`, then retry `docker pull`. A public image does not require Docker Hub login. [Docker login](https://docs.docker.com/reference/cli/docker/login/)
- **`insufficient_scope` in the publish workflow:** Check that the PAT has **Repo Read & Write** permission, `DOCKERHUB_USERNAME` is `nmp`, `DOCKERHUB_TOKEN` contains an active PAT, and the image name and tag are `nmp635/cdai-focal` and `latest`.
- **The workflow succeeded but the new tag is not visible:** Refresh the Docker Hub repository page and confirm that the workflow used `nmp635/cdai-focal:latest`.

## Technology and sources

- Web interface: Streamlit
- Model and pipeline: FOCAL, based on [tomoyoshki/focal](https://github.com/tomoyoshki/focal)
- Dataset: [nguyenmp/focal-dataset on Kaggle](https://www.kaggle.com/datasets/nguyenmp/focal-dataset)
