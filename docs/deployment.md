# Deployment and release artifacts

- [Public demo: Daibolical/inspection-desk](https://huggingface.co/spaces/Daibolical/inspection-desk)
- [Application source: Daiyan-Khan/inspection-desk](https://github.com/Daiyan-Khan/inspection-desk)
- [Initial static release snapshot](https://huggingface.co/spaces/Daibolical/inspection-desk/tree/43cbb952fea51beee295589ff0d5e94bf8d971e4): immutable Space commit `43cbb952fea51beee295589ff0d5e94bf8d971e4`, containing the built application, model weights, reference banks, public samples and frozen evidence.

The Space is a public, versioned artifact repository as well as the demo host. Successful publication is separate from functional checks of inference, persistence and exports on the hosted app.

## Hugging Face Static Space

1. Complete data preparation, artifact generation, evaluation, tests and the production build.
2. Run `npm run package:space`. It checks required outputs and copies the production build plus licence notices to `.publish/space/`.
3. Use the public [Daibolical/inspection-desk Space](https://huggingface.co/spaces/Daibolical/inspection-desk), configured as **Static**. No paid compute is needed.
4. Upload the contents of `.publish/space/`, preserving its directory structure. The packaged README declares `sdk: static` and `app_file: index.html`; it does not run a build on the host.
5. Open the resulting Space and verify a real inference, a saved/reloaded review, the evaluation page and exports. Do not call deployment complete until these work at the public URL.

The production files include pinned model and reference artifacts in the public Space repository. Each result also carries its artifact version. Replace the full packaged build together when updating an experiment; never update a bank or threshold independently.

## Public GitHub repository

The source repository is [Daiyan-Khan/inspection-desk](https://github.com/Daiyan-Khan/inspection-desk). Review the staged file list before pushing updates. Include source, package lock, tests, documentation, curated samples and frozen evaluation artifacts; do not include raw data, virtual environments, caches, private research, secrets or installed dependencies.

The project’s original model file is excluded from GitHub and regenerated from its pinned source. The immutable Space snapshot above preserves the exact deployed model and its accompanying artifacts. Repository line-ending rules preserve checksum-validated text across Windows and Linux checkouts.

## Repeatability and updates

- Commit `package-lock.json` and use `npm ci`.
- Keep the dataset source revision, archive identifier, file hashes, split manifest and duplicate audit.
- A scientific method change requires a new experiment version. The already inspected test set can no longer be described as untouched for subsequent model selection.
- The public demo uses static files and local inference. Hosting availability and visitor-device performance are separate limitations.

## Access

Publishing requires the repository owner's authenticated GitHub and Hugging Face accounts. Never place access tokens in application code, environment variables exposed to the frontend, screenshots, documentation or chat. The deployed static app needs no credential.
