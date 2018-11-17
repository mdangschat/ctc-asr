"""
Collection of TensorFlow hooks.
"""

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
                 log_every_n_steps=None,
                 log_every_n_secs=None,
                 query_every_n_steps=None,
                 query_every_n_secs=None,
                 output_dir=None,
                 summary_writer=None,
                 stats=('mem_used', 'mem_free', 'mem_total', 'mem_util', 'gpu_util'),
                 average_n=1,
                 suppress_stdout=False,
                 group_tag='gpu'):
        """
        Create an instance of `GPUStatisticsHook`.

        Arguments:
            log_every_n_steps (int):
                Integer controlling after how many (global) steps the hook is supposed to log the
                averaged values to tensorboard or the logging stream.
                When set `every_n_secs` must be None.
            log_every_n_secs (int):
                Integer controlling after how many seconds the hook is supposed to log the
                averaged values to tensorboard or the logging stream.
                When set `every_n_steps` must be None.
            query_every_n_steps (int):
                Integer controlling after how many (global) steps the hook is supposed to query
                values from the hardware.
                When set `every_n_secs` must be None.
            query_every_n_secs (int):
                Integer controlling after how many seconds the hook is supposed to query
                values from the hardware.
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
            average_n (int):
                Integer controlling how many values (i.e. results of a query) should be memorized
                for averaging.
                Default is 1, resulting in only the value from the last query execution
                being remembered.
            suppress_stdout (bool):
                If True, statistics are only logged to tensorboard.
                If False, statistics are logged to tensorboard and are written into tensorflow
                logging with INFO level.
            group_tag (str):
                Name of the tag under which the values will appear in tensorboard.
                Default is 'gpu'
        """

        # Check if only log_every_n_steps or only log_every_n_secs is set.
        if (log_every_n_steps is None) == (log_every_n_secs is None):
            raise ValueError("exactly one of log_every_n_steps and log_every_n_secs should be "
                             "provided.")

        # Check if only query_every_n_steps or only query_every_n_secs is set.
        if (query_every_n_steps is None) == (query_every_n_secs is None):
            raise ValueError("exactly one of query_every_n_steps and query_every_n_secs should be "
                             "provided.")

        # Timer controlling how often the statistics are queried from the GPUs.
        self._query_timer = tf.train.SecondOrStepTimer(every_steps=query_every_n_steps,
                                                       every_secs=query_every_n_secs)

        # Timer controlling how often statistics are logged (i.e. written to TB or to logging).
        self._log_timer = tf.train.SecondOrStepTimer(every_steps=log_every_n_steps,
                                                     every_secs=log_every_n_secs)

        # Initialize the internal variables.
        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._last_global_step = None
        self._global_step_check_count = 0
        self._steps_per_run = 1
        self._global_step_tensor = None
        self._statistics_to_log = stats
        self._suppress_stdout = suppress_stdout
        self._group_tag = group_tag

        self._average_n = average_n
        self._gpu_statistics = dict()

        self._global_step_write_count = 0

        # Initialize the NVML interface.
        nvml.nvmlInit()

        # Query the number of available GPUs.
        self._deviceCount = nvml.nvmlDeviceGetCount()

        # Create a summary dict for each GPU.
        for gpu_id in range(self._deviceCount):
            self._gpu_statistics[gpu_id] = self.__init_gpu_summaries()

    # def _set_steps_per_run(self, steps_per_run):
    #     self._steps_per_run = steps_per_run

    @staticmethod
    def __statistic_keys():
        """
        Get the keys for all statistics that the hook can query.

        Returns:
            list:
            List of keys.
        """
        return [
            'mem_used',  # Used memory.
            'mem_free',  # Free memory.
            'mem_total',  # Total memory.
            'mem_util',  # Memory IO utilization.
            'gpu_util'  # GPU utilization.
        ]

    def __init_gpu_summaries(self):
        """
        Create a dictionary with all summary keys initialized as empty lists.

        Returns:
            dict:
            Dictionary containing an empty list for each key from `__statistic_keys`.
        """
        summaries = dict()
        for key in self.__statistic_keys():
            summaries[key] = list()

        return summaries

    def __query_mem(self, handle):
        """
        Query information on the memory of a GPU.

        Arguments:
            handle:
                NVML device handle.

        Returns:
            summaries (:obj:`dict`):
                Dictionary containing the memory values for ['mem_used', 'mem_free', 'mem_total'].
                All values are given in MiB as integers.
        """
        # Query information on the GPUs memory usage.
        info = nvml.nvmlDeviceGetMemoryInfo(handle)

        summaries = dict()
        bytes_mib = 1024.0 ** 2
        summaries['mem_used'] = int(info.used / bytes_mib)
        summaries['mem_free'] = int(info.free / bytes_mib)
        summaries['mem_total'] = int(info.total / bytes_mib)

        return summaries

    def __query_util(self, handle):
        """
        Query information on the utilization of a GPU.

        Arguments:
            handle:
                NVML device handle.

        Returns:
            summaries (:obj:`dict`):
                Dictionary containing the memory values for ['mem_util', 'gpu_util'].
                All values are given as integers  in the range (0, 100).
        """
        # Query information on the GPU utilization.
        util = nvml.nvmlDeviceGetUtilizationRates(handle)

        summaries = dict()
        # Percent of time over the past second during which global (device) memory was being
        # read or written.
        summaries['mem_util'] = util.memory
        # Percent of time over the past second during which one or more kernels was executing
        # on the GPU.
        summaries['gpu_util'] = util.gpu

        return summaries

    def begin(self):
        """
        Is called once before the default graph in the active tensorflow session is
        finalized and the training has starts.
        The hook can modify the graph by adding new operations to it.
        After the begin() call the graph will be finalized and the other callbacks can not modify
        the graph anymore. Second call of begin() on the same graph, should not change the graph.
        """
        # Create a summary writer if possible.
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = summary_io.SummaryWriterCache.get(self._output_dir)

        # Get read access to the global step tensor.
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use StepCounterHook.")

    def end(self, session):
        """
        Called at the end of a session.

        Arguments:
            session (tf.Session):
                The `session` argument can be used in case the hook wants to run final ops,
                such as saving a last checkpoint.
        """
        # Shutdown the NVML interface.
        nvml.nvmlShutdown()

    def before_run(self, run_context):
        """
        Is called once before each call to session.run (training iteration in general).
        At this point the graph is finalized and you can not add ops.

        Arguments:
            run_context (tf.train.SessionRunContext):
                The `run_context` argument is a `SessionRunContext` that provides
                information about the upcoming `run()` call: the originally requested
                op/tensors, the TensorFlow Session.
                SessionRunHook objects can stop the loop by calling `request_stop()` of
                `run_context`.
                Sadly you have to take a look at 'tensorflow/python/training/session_run_hook.py'
                for more details.
        Returns:
            tf.train.SessionRunArgs:
                None or a `SessionRunArgs` object.
                Represents arguments to be added to a `Session.run()` call.
                Sadly you have to take a look at 'tensorflow/python/training/session_run_hook.py'
                for more details.
        """
        # Request to read the global step tensor when running the hook.
        # The content of the requested tensors is passed to the hooks `after_run` function.
        fetches = [
            # This will deliver the global step as it was before the `session.run`
            # call was executed.
            self._global_step_tensor
        ]
        return session_run_hook.SessionRunArgs(fetches=fetches)

    def after_run(self, run_context, run_values):
        """
        Is called once after each call to session.run (training iteration in general).
        At this point the graph is finalized and you can not add ops.

        Arguments:
            run_context (tf.train.SessionRunContext):
                The `run_context` argument is a `SessionRunContext` that provides
                information about the upcoming `run()` call: the originally requested
                op/tensors, the TensorFlow Session.
                SessionRunHook objects can stop the loop by calling `request_stop()` of
                `run_context`.
                Sadly you have to take a look at 'tensorflow/python/training/session_run_hook.py'
                for more details.
            run_values (tf.train.SessionRunValues):
                Contains the results of `Session.run()`
                However, this only seems to contain the results for the operations requested with
                the `before_run`.
                Sadly you have to take a look at 'tensorflow/python/training/session_run_hook.py'
                for more details.
        """
        # Ignore input argument.
        _ = run_context

        # Get the values of the tensors requested inside the `before_run` function.
        # read the global step as it was before the `session.run` call was executed.
        stale_global_step = run_values.results[0]

        # Check if the query timer should trigger for the current global step (i.e. last step + 1).
        if self._query_timer.should_trigger_for_step(stale_global_step + self._steps_per_run):
            # Get the actual global step from the global steps tensor.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._query_timer.should_trigger_for_step(global_step):
                # Get the elapsed time and elapsed steps since the last trigger event.
                elapsed_time, elapsed_steps = self._query_timer.update_last_triggered_step(
                    global_step)
                if elapsed_time is not None:
                    self._update_statistics(elapsed_steps, elapsed_time, global_step)

        # Check if the log timer should trigger for the current global step (i.e. last step + 1).
        if self._log_timer.should_trigger_for_step(stale_global_step + self._steps_per_run):
            # Get the actual global step from the global steps tensor.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._log_timer.should_trigger_for_step(global_step):
                # Get the elapsed time and elapsed steps since the last trigger event.
                elapsed_time, elapsed_steps = self._log_timer.update_last_triggered_step(
                    global_step)
                if elapsed_time is not None:
                    self._log_statistics(elapsed_steps, elapsed_time, global_step)

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

    def _update_statistics(self, elapsed_steps, elapsed_time, global_step):
        """
        Collect and store all summary values.

        Arguments:
            elapsed_steps (int):
                The number of steps between the current trigger event and the last one.
            elapsed_time (float):
                The number of seconds between the current trigger event and the last one.
            global_step (tf.Tensor):
                Global step tensor.
        """
        # Iterate the available GPUs.
        for gpu_id in range(self._deviceCount):
            summaries = dict()

            # Acquire a GPU device handle.
            handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)

            # Query information on the GPUs memory usage.
            summaries.update(self.__query_mem(handle))

            # Query information on the GPUs utilization.
            summaries.update(self.__query_util(handle))

            # Update the value history for the current GPU.
            for k in summaries.keys():
                if k in self._statistics_to_log:
                    self._gpu_statistics[gpu_id][k] = \
                        self._gpu_statistics[gpu_id][k][-self._average_n:] + [summaries[k]]

    def _log_statistics(self, elapsed_steps, elapsed_time, global_step):
        """
        Collect and store all summary values.

        Arguments:
            elapsed_steps (int):
                The number of steps between the current trigger event and the last one.
            elapsed_time (float):
                The number of seconds between the current trigger event and the last one.
            global_step (tf.Tensor):
                Global step tensor.
        """

        # Write summary for tensorboard.
        if self._summary_writer is not None:
            summary_list = list()
            # Add only summaries.
            for gpu_id in self._gpu_statistics.keys():
                for statistic in self._gpu_statistics[gpu_id].keys():
                    # only add them if they are requested for logging.
                    if statistic in self._statistics_to_log:
                        values = self._gpu_statistics[gpu_id][statistic]
                        # Only Calculate and write average if there is data available.
                        if len(values) > 0:
                            avg_value = sum(values) / len(values)
                            avg_summary = Summary.Value(tag='{}/{}:{}'
                                                        .format(self._group_tag, gpu_id, statistic),
                                                        simple_value=avg_value)
                            summary_list.append(avg_summary)

            # Write all statistics as simple scalar summaries.
            summary = Summary(value=summary_list)
            self._summary_writer.add_summary(summary, global_step)

        # Log summaries to the logging stream.
        if not self._suppress_stdout:
            for gpu_id in self._gpu_statistics.keys():
                # Acquire a GPU device handle.
                handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)

                # Query the device name.
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')

                for statistic in self._gpu_statistics[gpu_id].keys():
                    # Log utilization information with INFO level.
                    logging.info("%s: %s", name, '{}: {}'
                                 .format(statistic, self._gpu_statistics[gpu_id][statistic]))


# The following code has been inspired by <https://stackoverflow.com/a/45681782>:
class TraceHook(tf.train.SessionRunHook):
    """
    Hook to perform Traces every N steps.
    """

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
    """
    Log loss and runtime.
    """

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
