# Running on Unity (UMass, MGHPCC)

Step-by-step for launching fedcast-workflow by hand on the
[Unity](https://unity.rc.umass.edu) Slurm cluster. Everything runs from a
Unity **login node**: HTCondor's DAGMan runs there and hands each job to
Slurm through BLAHP. Placeholders: `<user>` is your Unity username,
`<pi_account>` your Slurm account (`sacctmgr show assoc user=$USER
format=account`, usually `pi_<pi-username>`).

## 1. SSH access

Unity takes SSH keys registered in the account portal
(<https://account.unityhpc.org>, Account Settings → SSH keys). On your own
machine:

```sh
ssh-keygen -t ed25519 -f ~/.ssh/unity_ed25519      # paste the .pub into the portal
cat >> ~/.ssh/config <<'EOF'
Host unity
    HostName unity.rc.umass.edu
    User <user>
    IdentityFile ~/.ssh/unity_ed25519
    IdentitiesOnly yes
EOF
ssh unity hostname
```

**Always come back to the same login node.** `unity.rc.umass.edu` rotates
between login nodes, but a workflow's HTCondor queue (DAGMan) lives only on
the node you submitted from: from any other node `pegasus-status` and
`condor_q` see nothing and `pegasus-remove` cannot reach the run. Login
nodes are internal names, so jump to yours through the public one, by
adding to `~/.ssh/config`:

```
Host unity7
    HostName login7.unity.rc.umass.edu      # the node you submit from
    User <user>
    IdentityFile ~/.ssh/unity_ed25519
    IdentitiesOnly yes
    ProxyJump unity
```

and then `ssh unity7` for everything below.

## 2. One-time setup on Unity

Unity has no Pegasus module, and the system Python lacks `ensurepip`, so
Pegasus 6.0.0.dev0 (needed for job tags and hosted site catalogs) goes into
a `uv` virtualenv. The `pegasus_wms` wheel is a full install —
`pegasus-plan`, `pegasus-run`, `pegasus-status` and the rest.

```sh
git clone https://github.com/pegasus-isi/fedcast-workflow.git ~/fedcast-workflow
cd ~/fedcast-workflow

module load uv/latest python/3.12.3
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install https://download.pegasus.isi.edu/pegasus/6.0.0.dev0/pegasus_wms-6.0.0.dev0-py3-none-manylinux_2_34_x86_64.whl
uv pip install -r requirements.txt pythonsed     # pythonsed: runtime doubling on retry

# Hosted Unity site catalog, plus your account and an A100 pin on GPU jobs
echo "pegasus.catalog.site.repo.file = unity.yml" >> ~/.pegasusrc
curl -O https://raw.githubusercontent.com/pegasushub/pegasus-site-catalogs/main/conf/unity.yml
./custom_sites.py --style slurm --base unity.yml --project <pi_account> \
    --gpu pegasus:glite.arguments=--constraint=a100
```

The paper used one A100 per job, so pin A100s for timing and full runs —
its GPU-hour figures assume them. **Leave the pin off for the pilot**
(drop the `--gpu` argument): it trains a 128x128 model at batch size 1 and
runs on any GPU, while A100 jobs can queue for hours (on 2026-10-06, 181 of
328 pending `gpu` jobs wanted the 8 A100 nodes). Other constraints Unity
offers include `a100-80g`, `h100`, `l40s` and `vram48`
(`sinfo -p gpu -o "%f"`). The pin lives in `sites.yml`: re-run
`custom_sites.py`, then the generator, then plan.

### Container images

Put `FedCast_data.sif`, `FedCast_train.sif` and `FedCast_eval.sif` in
`Apptainer/`. The images hold only the software stack — every `bin/`
script is staged from your checkout at run time — so an image only needs
rebuilding when its `.def` changes.

Unprivileged builds on Unity are partly working (2026-10-06). The data image
builds on a compute node in about a minute:

```sh
srun -p cpu -A <pi_account> -t 60 -c 4 --mem=16G \
    apptainer build --fakeroot Apptainer/FedCast_data.sif Apptainer/FedCast_data.def
```

The train and eval images (on the `pytorch/pytorch` base) fail in `%post`
under `--fakeroot`, so for now copy those two from a host that has them. Copying through a laptop is slow
(~2 MB/s seen); host to host is much faster:

```sh
# from the host holding the images, with your Unity key forwarded (ssh -A)
scp -4 Apptainer/FedCast_*.sif <user>@unity.rc.umass.edu:fedcast-workflow/Apptainer/
```

`-4` matters if that host prefers IPv6: Unity closed IPv6 SSH connections in
testing. Compare `sha256sum` on both ends after the copy.

## 3. Each run

```sh
ssh unity
cd ~/fedcast-workflow && source .venv/bin/activate
git pull                     # pick up workflow changes; re-run the generator after

# Pilot: 2 sites, 1 month, 2 FL rounds — always run this first on a new setup
python workflow_generator.py --test --shared-filesystem

# Full E1 reproduction
# python workflow_generator.py --start-month 2020-11 --months 48 --shared-filesystem

pegasus-plan --submit -s compute --cleanup leaf --output-dir output workflow.yml
```

Use the `pegasus-plan` line the generator prints. On Unity it always
includes `--cleanup leaf`, because the FL rounds are deferred sub-workflows
and per-file cleanup cannot plan on a site that stages through its own
scratch (README, "Cleanup on a batch site").

## 4. Monitoring

```sh
pegasus-status -l <run-dir>     # run-dir is printed by pegasus-plan, e.g.
                                # ~/fedcast-workflow/<user>/pegasus/fedcast/run0001
squeue -u $USER                 # the Slurm jobs Pegasus submitted
pegasus-analyzer <run-dir>      # after a failure: the failing job's output
pegasus-remove <run-dir>        # stop a run
pegasus-run <run-dir>           # resume a stopped or failed run where it left off
```

Results are staged to `~/fedcast-workflow/output/`; start with
`validation_report.md` (its Benchmark line should read 12/12).

### Slurm commands

Pegasus submits each job to Slurm under a name derived from the job
(`fltrainclienttr`, `traindgmrtrainc`, ...), so the usual Slurm tools work
directly. Use them to look; stop a whole run with `pegasus-remove`, not
`scancel` — a cancelled job just looks like a failure to Pegasus and gets
retried.

```sh
# Your jobs: state, run time, and why pending ones are waiting
squeue -u $USER
squeue -u $USER -o "%.10i %.9P %.18j %.8T %.10M %.20S %R %f"
#   %S = estimated start (N/A if Slurm cannot predict), %R = reason,
#   %f = features asked for (e.g. a100)
squeue -u $USER --start             # estimated start times only

# One job in detail (partition, account, GPUs, constraint, time limit)
scontrol show job <jobid>

# Finished jobs: did it succeed, how long, how much memory, which node
sacct -u $USER -S today -o JobID,JobName%20,Partition,State,Elapsed,MaxRSS,NodeList
sacct -j <jobid> -o JobID,State,ExitCode,Elapsed,AllocTRES%60

# Why you are behind in line
sprio -u $USER                      # priority breakdown of pending jobs
sshare -U -u $USER                  # fairshare (lower = more recent usage)

# Cluster capacity
sinfo -p gpu,cpu -s                 # nodes idle/allocated per partition
sinfo -p gpu -N -o "%N %T %f %G" | grep a100     # A100 nodes and state
squeue -p gpu -t PD | wc -l         # how many jobs wait for the gpu partition

# Your account and limits
sacctmgr show assoc user=$USER format=account,partition,maxjobs,grptres

# Cancel a single stray job (not one that belongs to a running workflow)
scancel <jobid>
```

Common `REASON` values in `squeue`: `Priority` (others are ahead of you),
`Resources` (you are next, waiting for a node to free up),
`QOSMaxJobsPerUserLimit` / `AssocGrpGRES` (you have hit a per-user or group
limit), `ReqNodeNotAvail` (a constraint no available node meets, often
during maintenance).

## 5. Where the requests end up

Each job carries `pegasus.project` (→ `#SBATCH --account`), the partition
(`cpu`, or `gpu` for tagged jobs), its runtime (→ `--time`, doubled on
retry), cores, memory, and on GPU jobs `--gpus=1`, the `--constraint` from
`sites.yml`, and Apptainer `--nv`. To confirm before submitting, plan without
`--submit` and read a GPU job's `.sub` file in the run directory:
`pegasus_queue`, `pegasus_project`, `pegasus_gpus` and
`pegasus_glite_arguments` should all be set.

## 6. Storage

Home is 100 GB — enough for pilots. A full 48-month run stages ~47 GB of
crops per CONUS fetch job, so run it from a checkout under
`/work/pi_<pi-username>` or a `/scratch3` workspace. A scratch directory
outside the workflow directory must also be bound into the containers with
`--container-bind <dir>` (next section).

## 7. Troubleshooting

**Every job exits 127, kickstart says "Unable to execute the specified
binary".** Pegasus staged inputs as symlinks (`pegasus.transfer.links`) into
the workflow directory, and PegasusLite starts containers with `--no-home`,
so the links dangle inside the container. The generator binds the workflow
and output directories automatically on sites that stage through their own
filesystem (Unity) or with `--shared-filesystem`; add `--container-bind
<dir>` for any other directory inputs may live in, then regenerate and
replan.

**A job fails with only an exit code and no error message.**
pegasus-kickstart keeps just the first line of a job's stderr once
Lightning has written to it, so tracebacks used to vanish. The wrappers now
mirror ERROR log lines and uncaught tracebacks to stdout, which the job
record keeps whole: look in the `stdout:` block of the job's `.out.NNN`
file (or `pegasus-analyzer -v`).

**Training fails: "You set `--ntasks=2` in your SLURM bash script".**
Lightning noticed the `SLURM_*` variables and switched to its Slurm
multi-process mode; Pegasus's `cores` reach Slurm as `--ntasks`, which that
mode rejects. Fixed in `fedcast_common.fit_one_epoch` by pinning Lightning
to `LightningEnvironment` (every job is one process on one GPU). A manual
`srun -c 2` test will not reproduce it — that is one task with two CPUs;
use `-n 2`.

**`srun` with more than one GPU is refused.** Unity requires `-N 1` (or
`--constraint=mpi`) whenever a job asks for more than one GPU.

**Binaries will not run from `/tmp` on a login node.** It is mounted
`noexec`; use a directory under your home.

**`python3 -m venv` fails with an ensurepip error.** Use the `uv` steps
above; the system Python has no `ensurepip`.

**Jobs pend for a long time.** `squeue -u $USER` shows the reason;
`(Priority)` with no start estimate means the constraint's nodes are
contended and your fairshare is low (`sshare -U -u $USER`). Jobs already
planned keep their constraint, including evaluation jobs that have not
reached Slurm yet, so changing `sites.yml` mid-run is not enough:
`pegasus-remove` the run, re-run `custom_sites.py` and the generator, and
plan again.

**`pegasus-status` says Idle but the job is running.** HTCondor hands
each job to Slurm and polls Slurm for its state every minute or two, so
`pegasus-status` and `condor_q` lag. `squeue -u $USER` is the real state:
a job RUNNING there is running, whatever Pegasus shows.

**A benchmark event logs "init shifted".** The MRMS archive has a gap in
that event's default sample; `fetch_benchmark` moved its initialization to
the nearest complete sample and recorded the shift (`init_shift_s`). Known
case: 20220307_2120, +14 min.
