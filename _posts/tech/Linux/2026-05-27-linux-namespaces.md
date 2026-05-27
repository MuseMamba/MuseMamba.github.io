---
layout: post
title: "Linux Namespaces: From First Principles to Kernel Implementation"
category: 技术
tags: [Linux, Namespace, Kernel, Container]
keywords: ["Linux Namespace", "Container Isolation", "clone", "unshare", "setns", "nsproxy"]
description: "A first-principles walkthrough of Linux namespaces — the 8 types, userspace APIs, and kernel implementation details."
---

# Linux Namespaces: From First Principles to Kernel Implementation

## TL;DR

- **Namespaces** partition global kernel resources so that each group of processes sees its own isolated instance of that resource.
- Linux provides **8 namespace types**: Mount, UTS, IPC, PID, Network, User, Cgroup, and Time.
- Userspace controls namespaces through three syscalls: `clone()`, `unshare()`, and `setns()`.
- In the kernel, each process holds a pointer to a `struct nsproxy`, which aggregates references to the per-type namespace objects.
- Namespaces are the foundational building block of Linux containers (Docker, Podman, LXC).

## First Principles

Traditional Unix assumes a single, shared view of system resources: one hostname, one process ID tree, one set of mount points, one network stack. Every process competes for and observes the same global state.

The **isolation problem**: how do you give a process (or group of processes) an independent view of a global resource without duplicating the entire OS?

The answer is **namespaces** — a lightweight virtualization mechanism that wraps a global resource in an abstraction layer. Processes inside a namespace see their own isolated instance; processes outside see theirs. The kernel multiplexes the underlying resource.

Key design principles:
- **Minimal overhead** — namespaces are thin wrappers, not full VMs.
- **Composable** — each resource type is isolated independently; you mix and match.
- **Hierarchical** — some namespaces (PID, User) form parent-child trees.
- **Inherited by default** — child processes share their parent's namespaces unless explicitly given new ones.

## The 8 Namespace Types

| Type    | Flag             | Isolates                          | Since   |
|:--------|:-----------------|:----------------------------------|:--------|
| Mount   | `CLONE_NEWNS`    | Filesystem mount points           | 2.4.19  |
| UTS     | `CLONE_NEWUTS`   | Hostname and NIS domain name      | 2.6.19  |
| IPC     | `CLONE_NEWIPC`   | SysV IPC, POSIX message queues    | 2.6.19  |
| PID     | `CLONE_NEWPID`   | Process ID number space           | 2.6.24  |
| Network | `CLONE_NEWNET`   | Network devices, stacks, ports    | 2.6.29  |
| User    | `CLONE_NEWUSER`  | UIDs, GIDs, capabilities          | 3.8     |
| Cgroup  | `CLONE_NEWCGROUP`| Cgroup root directory view        | 4.6     |
| Time    | `CLONE_NEWTIME`  | CLOCK_MONOTONIC, CLOCK_BOOTTIME   | 5.6     |

### Mount Namespace (CLONE_NEWNS)

The first namespace type added to Linux (hence the generic `NEWNS` flag — "new namespace" before others existed).

Each mount namespace has its own mount tree. Changes (mount, unmount, bind-mount) inside one namespace are invisible to others.

```bash
# Create a new mount namespace and get a shell in it
unshare --mount /bin/bash

# Mounts here are private to this namespace
mount -t tmpfs tmpfs /mnt
ls /mnt   # visible here
# From another terminal: /mnt is unchanged
```

**Use case**: Containers get their own root filesystem via pivot_root + mount namespace.

### UTS Namespace (CLONE_NEWUTS)

Isolates the hostname (`uname -n`) and NIS domain name. UTS stands for "UNIX Time-sharing System" — the `utsname` struct in the kernel.

```bash
unshare --uts /bin/bash
hostname container-alpha
hostname   # → container-alpha
# Host still shows the original hostname
```

**Use case**: Each container reports its own hostname.

### IPC Namespace (CLONE_NEWIPC)

Isolates System V IPC objects (shared memory segments, semaphore sets, message queues) and POSIX message queues.

Processes in different IPC namespaces cannot access each other's IPC objects even if they use the same key.

```bash
unshare --ipc /bin/bash
ipcmk -M 1024   # create shared memory visible only in this namespace
ipcs             # shows the segment
# From host: ipcs shows nothing new
```

**Use case**: Prevent cross-container IPC side channels.

### PID Namespace (CLONE_NEWPID)

Creates an independent process ID number space. The first process in a new PID namespace becomes PID 1 (the init of that namespace).

PID namespaces are **hierarchical**: a parent namespace can see child PIDs (mapped to a different number), but children cannot see parent PIDs.

```bash
unshare --pid --fork --mount-proc /bin/bash
ps aux   # only shows processes in this namespace
echo $$  # 1
```

**Key properties**:
- PID 1 in the namespace receives orphaned children (init reaping).
- If PID 1 dies, the kernel sends SIGKILL to all other processes in that namespace.
- Processes have different PIDs in each namespace they're visible in.

### Network Namespace (CLONE_NEWNET)

Each network namespace gets its own:
- Network devices (including loopback)
- IPv4/IPv6 protocol stacks
- Routing tables
- Firewall rules (iptables/nftables)
- Sockets and port space

```bash
# Create a named network namespace
ip netns add myns

# Run a command inside it
ip netns exec myns ip link list
# → only lo, which is DOWN by default

# Connect namespaces with a veth pair
ip link add veth0 type veth peer name veth1
ip link set veth1 netns myns
ip addr add 10.0.0.1/24 dev veth0
ip netns exec myns ip addr add 10.0.0.2/24 dev veth1
ip link set veth0 up
ip netns exec myns ip link set veth1 up

# Now 10.0.0.1 <-> 10.0.0.2 can communicate
```

**Use case**: Container networking — each container gets its own IP, ports, and routing.

### User Namespace (CLONE_NEWUSER)

Maps UIDs and GIDs between the namespace and the parent. A process can be root (UID 0) inside the namespace while being an unprivileged user outside.

```bash
unshare --user --map-root-user /bin/bash
id   # uid=0(root) gid=0(root)
# But outside, this process runs as your normal user
```

**Key properties**:
- The **only** namespace type that can be created without privileges.
- Capabilities are scoped to a user namespace — root inside has capabilities only within that namespace.
- User namespaces **own** other namespaces: creating a non-user namespace requires `CAP_SYS_ADMIN` in the owning user namespace.

UID/GID mapping is configured via `/proc/[pid]/uid_map` and `/proc/[pid]/gid_map`:

```
# Format: <ns_start> <host_start> <count>
0 1000 1       # UID 0 inside → UID 1000 outside
```

### Cgroup Namespace (CLONE_NEWCGROUP)

Virtualizes the view of `/proc/[pid]/cgroup` and the cgroup filesystem. Inside a cgroup namespace, the process sees its own cgroup as the root.

```bash
# Process in cgroup /docker/abc123 enters a cgroup namespace
# Inside, /proc/self/cgroup shows "/" instead of "/docker/abc123"
```

**Use case**: Prevent containers from discovering the host's cgroup hierarchy.

### Time Namespace (CLONE_NEWTIME)

The newest namespace type (Linux 5.6). Allows per-namespace offsets for `CLOCK_MONOTONIC` and `CLOCK_BOOTTIME`.

```bash
unshare --time /bin/bash
# Adjust the monotonic clock offset via /proc/[pid]/timens_offsets
# (must be set before any process in the namespace reads the clock)
```

**Use case**: Container migration/checkpoint-restore (CRIU) — restore a container on a different host without time jumping.

## Userspace APIs

### clone() — Create a new process in new namespaces

```c
#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>

int flags = CLONE_NEWPID | CLONE_NEWNS | CLONE_NEWNET | SIGCHLD;
pid_t pid = clone(child_fn, stack_top, flags, arg);
```

Each `CLONE_NEW*` flag creates a new namespace of that type for the child. Without the flag, the child shares the parent's namespace.

### unshare() — Move the calling process into new namespaces

```c
#include <sched.h>

// Current process leaves its mount and UTS namespaces
unshare(CLONE_NEWNS | CLONE_NEWUTS);
```

Unlike `clone()`, `unshare()` does not create a new process — it detaches the caller from the specified namespaces and creates fresh ones.

### setns() — Join an existing namespace

```c
#include <sched.h>
#include <fcntl.h>

int fd = open("/proc/1234/ns/net", O_RDONLY);
setns(fd, CLONE_NEWNET);  // caller joins PID 1234's network namespace
close(fd);
```

This is how `nsenter` and `docker exec` work — they open the target's namespace file and call `setns()`.

### /proc/[pid]/ns/ — Namespace handles

```bash
ls -la /proc/self/ns/
# lrwxrwxrwx 1 user user 0 ... cgroup -> 'cgroup:[4026531835]'
# lrwxrwxrwx 1 user user 0 ... ipc -> 'ipc:[4026531839]'
# lrwxrwxrwx 1 user user 0 ... mnt -> 'mnt:[4026531841]'
# lrwxrwxrwx 1 user user 0 ... net -> 'net:[4026531840]'
# lrwxrwxrwx 1 user user 0 ... pid -> 'pid:[4026531836]'
# lrwxrwxrwx 1 user user 0 ... time -> 'time:[4026531834]'
# lrwxrwxrwx 1 user user 0 ... user -> 'user:[4026531837]'
# lrwxrwxrwx 1 user user 0 ... uts -> 'uts:[4026531838]'
```

Each symlink target includes the inode number — two processes are in the same namespace if and only if the inode numbers match. Holding an open fd or a bind-mount to these files keeps the namespace alive even after all processes in it exit.

## Kernel Implementation

### struct nsproxy — The namespace aggregator

Every `task_struct` has a pointer to `nsproxy`, which groups all non-user namespaces:

```c
// include/linux/nsproxy.h
struct nsproxy {
    refcount_t count;
    struct uts_namespace   *uts_ns;
    struct ipc_namespace   *ipc_ns;
    struct mnt_namespace   *mnt_ns;
    struct pid_namespace   *pid_ns_for_children;
    struct net             *net_ns;
    struct time_namespace  *time_ns;
    struct time_namespace  *time_ns_for_children;
    struct cgroup_namespace *cgroup_ns;
};
```

Note: `user_ns` is **not** in `nsproxy` — it lives in `struct cred` because it's tied to the security context, not the resource context.

### struct ns_common — The per-type base

Every namespace struct embeds `ns_common`:

```c
// include/linux/ns_common.h
struct ns_common {
    struct dentry *stashed;   // for /proc/[pid]/ns/ dentry caching
    const struct proc_ns_operations *ops;
    unsigned int inum;        // inode number (unique namespace ID)
    refcount_t count;
};
```

The `ops` field provides type-specific callbacks:

```c
struct proc_ns_operations {
    const char *name;
    const char *real_ns_name;
    int type;                          // CLONE_NEW* flag
    struct ns_common *(*get)(struct task_struct *);
    void (*put)(struct ns_common *);
    int (*install)(struct nsset *, struct ns_common *);
    struct user_namespace *(*owner)(struct ns_common *);
    struct ns_common *(*get_parent)(struct ns_common *);
};
```

### Per-type namespace structs

Example — UTS namespace:

```c
// include/linux/utsname.h
struct uts_namespace {
    struct new_utsname name;    // hostname, domainname, etc.
    struct user_namespace *user_ns;
    struct ucounts *ucounts;
    struct ns_common ns;
};
```

Example — PID namespace:

```c
// include/linux/pid_namespace.h
struct pid_namespace {
    struct idr idr;                    // PID allocation
    struct rcu_head rcu;
    unsigned int pid_allocated;
    struct task_struct *child_reaper;   // namespace's init (PID 1)
    struct kmem_cache *pid_cachep;
    unsigned int level;                // nesting depth
    struct pid_namespace *parent;       // parent PID namespace
    struct user_namespace *user_ns;
    struct ucounts *ucounts;
    int reboot;
    struct ns_common ns;
};
```

### Namespace lifecycle: copy_namespaces()

When `fork()`/`clone()` is called, the kernel invokes `copy_namespaces()`:

```c
// kernel/nsproxy.c
int copy_namespaces(unsigned long flags, struct task_struct *tsk)
{
    struct nsproxy *old_ns = tsk->nsproxy;
    struct nsproxy *new_ns;

    // Fast path: no new namespace requested → share parent's nsproxy
    if (!(flags & (CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC |
                   CLONE_NEWPID | CLONE_NEWNET | CLONE_NEWCGROUP |
                   CLONE_NEWTIME))) {
        refcount_inc(&old_ns->count);
        return 0;
    }

    // Slow path: create new nsproxy and selectively copy/create namespaces
    new_ns = create_new_namespaces(flags, tsk, ...);
    tsk->nsproxy = new_ns;
    return 0;
}
```

For each `CLONE_NEW*` flag that is set, the kernel calls the corresponding `create_*_ns()` or `copy_*_ns()` function. If the flag is not set, the new `nsproxy` simply takes a reference to the existing namespace.

### create_new_namespaces() flow

```
create_new_namespaces(flags, tsk, ...)
 ├── copy_mnt_ns()       — if CLONE_NEWNS, duplicate mount tree
 ├── copy_utsname()      — if CLONE_NEWUTS, clone uts_namespace
 ├── copy_ipcs()         — if CLONE_NEWIPC, new IPC namespace
 ├── copy_pid_ns()       — if CLONE_NEWPID, new PID level
 ├── copy_net_ns()       — if CLONE_NEWNET, new network stack
 ├── copy_cgroup_ns()    — if CLONE_NEWCGROUP, new cgroup root view
 └── copy_time_ns()      — if CLONE_NEWTIME, new time offsets
```

### Reference counting and destruction

Namespaces are reference-counted via `ns_common.count`. When the last reference drops (all processes exited, no open fds, no bind-mounts), the type-specific destructor runs:

- UTS: `free_uts_ns()` — frees the `uts_namespace` struct
- PID: `free_pid_ns()` — tears down the PID idr, only when the last process in the deepest level exits
- Network: `net_drop_ns()` → `cleanup_net()` — unregisters devices, flushes routes, frees the entire `struct net`

### unshare_nsproxy_namespaces()

The `unshare()` syscall goes through:

```c
int unshare_nsproxy_namespaces(unsigned long flags, struct nsproxy **new_nsp, ...)
{
    // Creates a new nsproxy with fresh namespaces for the requested types
    *new_nsp = create_new_namespaces(flags, current, ...);
    return 0;
}
```

The caller then atomically swaps the current task's `nsproxy` pointer.

### setns() implementation

```c
// kernel/nsproxy.c  (simplified)
SYSCALL_DEFINE2(setns, int, fd, int, flags)
{
    struct nsset nsset = {};
    struct ns_common *ns = get_ns_from_fd(fd);  // resolve the namespace from fd

    // Validate: does the ns type match the flags?
    // Permission check: does caller have rights in the target user_ns?

    ns->ops->install(&nsset, ns);  // type-specific join logic
    commit_nsset(&nsset);          // swap nsproxy + creds atomically
    return 0;
}
```

## Namespaces and Containers

A container runtime (Docker, runc, crun) creates a container by combining namespaces:

```
Container = Mount NS + UTS NS + IPC NS + PID NS + Network NS + User NS + Cgroup NS
```

Typical `runc` flow:
1. `clone(CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWUTS | CLONE_NEWIPC | CLONE_NEWNET | CLONE_NEWCGROUP)`
2. In the child: `pivot_root()` into the container's rootfs
3. Configure the network namespace (veth pair, bridge)
4. Set hostname via `sethostname()`
5. Apply cgroup limits
6. Drop capabilities, apply seccomp filter
7. `exec()` the container's entrypoint

`docker exec` uses `setns()` to join the existing container's namespaces before exec'ing the user's command.

### Namespace composition diagram

```
Host Kernel
├── init_nsproxy (default namespaces for all host processes)
│
├── Container A
│   └── nsproxy_A → { mnt_ns_A, uts_ns_A, pid_ns_A, net_ns_A, ... }
│
├── Container B
│   └── nsproxy_B → { mnt_ns_B, uts_ns_B, pid_ns_B, net_ns_B, ... }
│
└── Container C (shares network with A)
    └── nsproxy_C → { mnt_ns_C, uts_ns_C, pid_ns_C, net_ns_A, ... }
                                                      ^^^^^^^^ shared
```

This composability is what makes namespaces powerful — you can share a network namespace between containers (pod networking in Kubernetes) while keeping everything else isolated.

## Case Study: How Claude Code Uses Namespaces

[Claude Code](https://code.claude.com), Anthropic's AI coding agent CLI, provides a concrete real-world example of using Linux namespaces for lightweight sandboxing — not to run full containers, but to **restrict what shell commands can access** at the OS level.

### The Problem

Claude Code executes arbitrary Bash commands on behalf of the user (builds, tests, scripts). Without isolation, a misbehaving or compromised command could:
- Write to sensitive files outside the project (e.g., `~/.ssh/`, `~/.bashrc`)
- Exfiltrate data over the network to unauthorized hosts
- Modify its own sandbox policy by editing settings files

### The Solution: Bubblewrap + User Namespaces

On Linux (and WSL2), Claude Code uses **[bubblewrap (bwrap)](https://github.com/containers/bubblewrap)** — the same tool Flatpak uses — to sandbox every Bash subprocess. Bubblewrap leverages **user namespaces** (`CLONE_NEWUSER`) as its entry point, which enables unprivileged sandboxing without root.

The isolation stack:

| Layer | Namespace / Mechanism | What It Restricts |
|:------|:---------------------|:------------------|
| Filesystem | **Mount namespace** (`CLONE_NEWNS`) | Bind-mounts only allowed paths into the sandbox; everything else is invisible or read-only |
| Network | Proxy + **network namespace** (`CLONE_NEWNET`) | All traffic is routed through a domain-allowlist proxy; unauthorized hosts are blocked |
| Process visibility | **PID namespace** (`CLONE_NEWPID`) | Sandboxed processes cannot see or signal host processes |
| Privileges | **User namespace** (`CLONE_NEWUSER`) | The sandbox runs as an unprivileged user mapping; no real root capabilities |

### How It Works in Practice

When Claude Code runs a sandboxed Bash command:

```
Claude Code (parent process)
│
├── Launches bwrap with:
│   ├── CLONE_NEWUSER  → unprivileged namespace (no real root)
│   ├── CLONE_NEWNS    → custom mount tree:
│   │   ├── bind-mount project dir (read-write)
│   │   ├── bind-mount /usr, /lib, etc. (read-only)
│   │   ├── deny write to ~/, /etc, /bin
│   │   └── fresh /tmp (isolated)
│   ├── CLONE_NEWPID   → separate PID space
│   └── Network proxy  → HTTP/SOCKS proxy enforcing domain allowlist
│
└── Child process (the user's command)
    └── Sees: only the project directory as writable,
              allowed network domains, isolated PID tree
```

### Filesystem Isolation via Mount Namespace

The mount namespace is the workhorse. Bubblewrap constructs a minimal mount tree:

- **Project directory**: bind-mounted read-write (the only writable location by default)
- **System libraries** (`/usr`, `/lib`, `/lib64`): bind-mounted read-only so commands can execute
- **Home directory** (`~/`): blocked or read-only (credentials like `~/.aws/`, `~/.ssh/` are hidden unless explicitly allowed)
- **Settings files**: always deny-write protected so sandboxed commands cannot modify their own policy

Additional write paths can be granted via `sandbox.filesystem.allowWrite`:

```json
{
  "sandbox": {
    "enabled": true,
    "filesystem": {
      "allowWrite": ["~/.kube", "/tmp/build"]
    }
  }
}
```

### Network Isolation

Rather than a full network namespace with no connectivity, Claude Code routes all sandboxed traffic through a **proxy server** running outside the sandbox:

- First connection to a new domain triggers a user prompt
- Approved domains are added to an allowlist
- The proxy filters by hostname (SNI-based, no TLS termination)
- `socat` relays traffic between the sandbox and the proxy

This is a pragmatic trade-off: full `CLONE_NEWNET` isolation would break all network tools, while the proxy approach allows controlled egress.

### The Ubuntu 24.04 Problem: AppArmor vs. User Namespaces

A real-world friction point: Ubuntu 24.04+ defaults `kernel.apparmor_restrict_unprivileged_userns=1`, which prevents bubblewrap from creating the user namespace it needs.

The fix is an AppArmor profile that grants `bwrap` the `userns` capability:

```bash
sudo tee /etc/apparmor.d/bwrap > /dev/null <<'EOF'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap>
}
EOF
sudo systemctl reload apparmor
```

This illustrates a tension in the namespace design: user namespaces were meant to enable unprivileged isolation, but distributions increasingly restrict them because they expand the kernel attack surface (every `CLONE_NEWUSER` call gives the process access to kernel code paths normally reserved for root).

### Why Not a Full Container?

Claude Code could run commands in Docker, but that would be heavyweight for an interactive CLI tool. The bubblewrap approach is:
- **Fast**: no daemon, no image pull, no layered filesystem — just `clone()` + bind-mounts
- **Composable**: filesystem and network restrictions are configured independently
- **Transparent**: commands behave normally; they just can't escape the boundary
- **Unprivileged**: no root or Docker daemon required (just user namespaces)

This is namespaces at their most surgical — using exactly the isolation primitives needed, nothing more.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│  Host (your machine)                                │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  Claude Code Process                          │  │
│  │  (user namespace owner, proxy server)         │  │
│  │                                               │  │
│  │  settings.json ──► sandbox policy             │  │
│  │       │                                       │  │
│  │       ▼                                       │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │  bwrap sandbox (per Bash command)       │  │  │
│  │  │                                         │  │  │
│  │  │  Mount NS: project/ (rw), /usr (ro)     │  │  │
│  │  │  PID NS:   isolated process tree        │  │  │
│  │  │  User NS:  unprivileged mapping         │  │  │
│  │  │  Network:  proxy → allowlist filter     │  │  │
│  │  │                                         │  │  │
│  │  │  $ npm test   ← runs here              │  │  │
│  │  │  $ git commit ← runs here              │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Key Takeaway

Claude Code's sandbox is a textbook example of **minimal namespace composition**: user namespaces for unprivileged entry, mount namespaces for filesystem boundaries, PID namespaces for process isolation, and a proxy-based network filter. It demonstrates that namespaces aren't just for full containerization — they're surgical tools for enforcing precise security boundaries in any application that spawns untrusted subprocesses.
