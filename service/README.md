# NVXSched Service

## XClient

`XCLI` is a command-line tool, it is provided to users to change the policy and give scheduling hints(e.g., priorities, utilization weights, deadlines) to the policy in `XServer`.

### Usage

You can run it by `./output/bin/xcli` in the root directory of NVXSched after compiling.

The parameters it accepts are as follows:

| Usage       | Type   | Description                   | Default   |
| ----------- | ------ | ----------------------------- | --------- |
| `-a,--addr` | string | NVXSched server ipv4 address. | 127.0.0.1 |
| `-p,--port` | int    | NVXSched server port.         | 50000     |

The subcommands it supports are as follows:

<table>
  <tr>
    <th align="center">Subcommand</th>
    <th align="center">Description</th>
    <th align="center">Usage</th>
    <th align="center">Type</th>
    <th align="center">Range</th>
  </tr>
  <tr>
    <td align="center">top</td>
    <td align="center">Show the top information of the XQueues</td>
    <td align="center" style="white-space:nowrap">-f,--frequency</td>
    <td align="center">float</td>
    <td align="center">[0.1,30]</td>
  </tr>
  <tr>
    <td align="center">list</td>
    <td align="center">List the information of the XQueues.</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">config</td>
    <td align="center" rowspan="4">Change the configuration of the XQueues.</td>
    <td align="center" style="white-space:nowrap">-x,--xqueue</td>
    <td align="center">uint64_t</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-l,--level</td>
    <td align="center" style="white-space:nowrap">int32_t</td>
    <td align="center">[1,3]</td>
  </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-t,--threshold</td>
    <td align="center">int64_t</td>
    <td align="center">[1,65536]</td>
  </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-b,--batch-size</td>
    <td align="center">int64_t</td>
    <td align="center">[1,65536]</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">policy</td>
    <td align="center" rowspan="2">Query or set the scheduler policy of the NVXSched server.</td>
    <td align="center">-q,--query</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-s -n,--set --name</td>
    <td align="center">string</td>
    <td align="center">XSCHED_POLICY</td>
   </tr>
   <tr>
    <td align="center" rowspan="5">hint</td>
    <td align="center" rowspan="5">Give a hint to the NVXSched server.</td>
    <td align="center">-x,--xqueue</td>
    <td align="center">uint64_t</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="center" style="white-space:nowrap">--pid</td>
    <td align="center">pid_t</td>
    <td align="center">-</td>
   </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-p,--priority</td>
    <td align="center">int32_t</td>
    <td align="center">[-255,255]</td>
   </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-u,--utilization</td>
    <td align="center">int32_t</td>
    <td align="center">[0,100]</td>
   </tr>
  <tr>
    <td align="center" style="white-space:nowrap">-t,--timeslice</td>
    <td align="center">int64_t</td>
    <td align="center">[100,10000000]</td>
   </tr>
</table>

### Examples

Use `top` to monitor activity or `list` to find the queue handles and process IDs used in later commands.
The following example selects `HPF` and sets the priority of a queue; replace `xqueue_handle` with the handle reported by `list`:

```bash
# Show the top information of the XQueues
./output/bin/xcli top -f 1
# List the information of the XQueues
./output/bin/xcli list
# Change the configuration of the XQueues
./output/bin/xcli config -x xqueue_handle -l 1 -t 16 -b 8
# Query the scheduler policy of the NVXSched server
./output/bin/xcli policy -q
# Set the scheduler policy of the NVXSched server
./output/bin/xcli policy -s -n HPF
# Give a hint to the NVXSched server
./output/bin/xcli hint -x xqueue_handle -p 10
```

Hints must match the selected policy.
For process scheduling with `PUP` or `SPUP`, use `--pid` to set a process weight and `-t` to set the average process timeslice in microseconds.
For example, the following commands select `SPUP`, give one process half the default weight of 100, and set the average timeslice to 2 ms:

```bash
./output/bin/xcli policy -s -n SPUP
./output/bin/xcli hint --pid pid -u 50
./output/bin/xcli hint -t 2000
```

Weights determine relative timeslices, so a weight of 50 gives that process half the base timeslice of a process with weight 100.
For queue scheduling with `UP`, use a queue handle with `-u`; its `-t` hint sets the budget for the whole scheduling round.
See the [hint reference](../sched/README.md#hint) for policy-specific meanings.

## XServer

`XServer` consists of following parts:

- `LocalScheduler` to execute processing logic.
- A server IPC channel to receive events and a channel for each client process to send operations.
- A http server to comminicate with `XCLI`.

XServer maintains a global mirror of each process's XQueue status (ready or idle, XPU device ID, process ID, etc.) and trigger the scheduling when receives events or policy timers expire.

### Run

You can run xserver by `./output/bin/xserver` in the root directory of NVXSched after compiling.

The default policy is `HPF`, default port is 50000. You can change them by parameters:

```bash
# Run XServer with default policy and port
./output/bin/xserver
# Run XServer with SPUP policy and port 50000
./output/bin/xserver SPUP 50000
```

After starting the server, set `XSCHED_SCHEDULER=GLB` and unset `XSCHED_POLICY` in each application to connect it to global scheduling.
Applications exchange scheduling events and operations with XServer through IPC; the HTTP port is used by XCLI.
See the [CUDA guide](../platforms/cuda/README.md#usage) for complete application configuration, including the prediction and driver settings for memory scheduling.
