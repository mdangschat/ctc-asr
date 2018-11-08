import time
from datetime import datetime
import pynvml as nvml
import tensorflow as tf

from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import summary_io, training_util, session_run_hook

from python.params import FLAGS


class GPUStatisticsHook(tf.train.SessionRunHook):
    """
    A session hook that log GPU statistics to tensorboard and to the log stream.
    """

    def __init__(self,
                 every_n_steps=None,
                 every_n_secs=None,
                 output_dir=None,
                 summary_writer=None,
                 stats=('mem_used', 'mem_free', 'mem_total', 'mem_util', 'gpu_util'),
                 suppress_stdout=False):
        """
        Create an instance of `GPUStatisticsHook`.

        Arguments:
            every_n_steps (int):
                Integer controlling after how many (global) steps the hook is executed.
                When set `every_n_secs` must be None.
            every_n_secs (int):
                Integer controlling after how many seconds the hook is executed.
                When set `every_n_steps` must be None.
            output_dir (str):
                In case `summary_writer` is None, this parameter is used to construct a
                FileWriter for writing a summary statistic.
            summary_writer (tensorflow.summary.FileWriter):
                FileWriter to use for writing the summary statistics.
            stats (:obj:`tuple` of `str`):
                List of strings to control what statistics are written to tensorboard.
                Valid strings are ('mem_used', 'mem_free', 'mem_total', 'mem_util', 'gpu_util').
                Note that ('mem_used', 'mem_free', 'mem_total') are logged in MiB and encompass
                the global GPU state (therefore including all processes running on that GPU).
                Note that ('mem_util', 'gpu_util') are given in percent (0, 100).
            suppress_stdout (bool):
                If True, statistics are only logged to tensorboard.
                If False, statistics are logged to tensorboard and are written into tensorflow
                logging with INFO level.
        """

        # Check if only every_n_steps or only every_n_secs is set.
        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError("exactly one of every_n_steps and every_n_secs should be provided.")

        # Create a timer that triggers either after time or after steps.
        self._timer = tf.train.SecondOrStepTimer(every_steps=every_n_steps, every_secs=every_n_secs)

        # Initialize the internal variables.
        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._last_global_step = None
        self._global_step_check_count = 0
        self._steps_per_run = 1
        self._global_step_tensor = None
        self._stats = stats
        self._suppress_stdout = suppress_stdout

        # Initialize the NVML interface.
        nvml.nvmlInit()

        # Query the number of available GPUs.
        self._deviceCount = nvml.nvmlDeviceGetCount()

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        # Create a summary writer if possible.
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = summary_io.SummaryWriterCache.get(self._output_dir)

        # Get read access to the global step tensor.
        # pylint: disable=protected-access
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use StepCounterHook.")

    def end(self, session):
        # Shutdown the NVML interface.
        nvml.nvmlShutdown()

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return session_run_hook.SessionRunArgs(self._global_step_tensor)

    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        # Iterate the available GPUs.
        for i in range(self._deviceCount):
            summaries = dict()

            # Acquire a GPU device handle.
            handle = nvml.nvmlDeviceGetHandleByIndex(i)

            # Query information on the GPUs memory usage.
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            summaries['mem_used'] = int(info.used / 1024.0 ** 2)
            summaries['mem_free'] = int(info.free / 1024.0 ** 2)
            summaries['mem_total'] = int(info.total / 1024.0 ** 2)

            # Query information on the GPUs utilization.
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            # Percent of time over the past second during which global (device) memory was being
            # read or written.
            summaries['mem_util'] = util.memory
            # Percent of time over the past second during which one or more kernels was executing
            # on the GPU.
            summaries['gpu_util'] = util.gpu

            # Define the tag name the gpu data will be shown under in tensorboard.
            gpu_tag = 'gpu:{}'.format(i)

            # Write summary for tensorboard.
            if self._summary_writer is not None:
                values = list()
                # Add only summaries that are requested for logging.
                for k in summaries.keys():
                    if k in self._stats:
                        s = Summary.Value(tag='{}/{}'.format(gpu_tag, k), simple_value=summaries[k])
                        values.append(s)

                # Write all statistics as simple scalar summaries.
                summary = Summary(value=values)
                self._summary_writer.add_summary(summary, global_step)

            if not self._suppress_stdout:
                # Query the device name.
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')

                # Log utilization information with INFO level.
                logging.info("%s (%s): %s", gpu_tag, name, 'gpu util: {} %, mem_io util: {} %'
                             .format(summaries['mem_util'], summaries['gpu_util']))

    def after_run(self, run_context, run_values):
        _ = run_context

        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(global_step)
                if elapsed_time is not None:
                    self._log_and_record(elapsed_steps, elapsed_time, global_step)

        # Check whether the global step has been increased. Here, we do not use the
        # timer.last_triggered_step as the timer might record a different global
        # step value such that the comparison could be unreliable. For simplicity,
        # we just compare the stale_global_step with previously recorded version.
        if stale_global_step == self._last_global_step:
            # Here, we use a counter to count how many times we have observed that the
            # global step has not been increased. For some Optimizers, the global step
            # is not increased each time by design. For example, SyncReplicaOptimizer
            # doesn't increase the global step in worker's main train step.
            self._global_step_check_count += 1
            if self._global_step_check_count % 20 == 0:
                self._global_step_check_count = 0
                logging.warning(
                    "It seems that global step (tf.train.get_global_step) has not "
                    "been increased. Current value (could be stable): %s vs previous "
                    "value: %s. You could increase the global step by passing "
                    "tf.train.get_global_step() to Optimizer.apply_gradients or "
                    "Optimizer.minimize.", stale_global_step, self._last_global_step)
        else:
            # Whenever we observe the increment, reset the counter.
            self._global_step_check_count = 0

        self._last_global_step = stale_global_step


# The following code has been inspired by <https://stackoverflow.com/a/45681782>:
class TraceHook(tf.train.SessionRunHook):
    """Hook to perform Traces every N steps."""

    def __init__(self, file_writer, log_frequency, trace_level=tf.RunOptions.FULL_TRACE):
        self._trace = log_frequency == 1
        self.writer = file_writer
        self.trace_level = trace_level
        self.log_frequency = log_frequency
        self._global_step_tensor = None

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use TraceHook.")

    def before_run(self, run_context):
        if self._trace:
            options = tf.RunOptions(trace_level=self.trace_level)
        else:
            options = None

        return tf.train.SessionRunArgs(fetches=self._global_step_tensor, options=options)

    def after_run(self, run_context, run_values):
        global_step = run_values.results - 1
        if self._trace:
            self._trace = False
            self.writer.add_run_metadata(run_values.run_metadata, '{}'.format(global_step))
        if not (global_step + 1) % self.log_frequency:
            self._trace = True


class LoggerHook(tf.train.SessionRunHook):
    """Log loss and runtime."""

    def __init__(self, loss_op):
        self.loss_op = loss_op
        self._global_step_tensor = None
        self._start_time = 0

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        self._start_time = time.time()

    def before_run(self, run_context):
        # Asks for loss value and global step.
        return tf.train.SessionRunArgs(fetches=[self.loss_op, self._global_step_tensor])

    def after_run(self, run_context, run_values):
        loss_value, global_step = run_values.results

        if global_step % FLAGS.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
            sec_per_batch = duration / float(FLAGS.log_frequency)
            batch_per_sec = float(FLAGS.log_frequency) / duration

            print('{:%Y-%m-%d %H:%M:%S}: Epoch {:,d} (step={:,d}); loss={:.4f}; '
                  '{:.1f} examples/sec ({:.3f} sec/batch) ({:.2f} batch/sec)'
                  .format(datetime.now(),
                          global_step // (FLAGS.num_examples_train // FLAGS.batch_size - 1),
                          global_step, loss_value, examples_per_sec,
                          sec_per_batch, batch_per_sec))
