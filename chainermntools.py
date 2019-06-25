import chainer.datasets
import numpy


def scatter_dataset_no_comm(dataset, comm, shuffle=False, seed=0):
    """Scatter the given dataset to the workers in the communicator.

    This function does not use MPI communication.

    The dataset of every worker has to be the same (assuming file sharing
    system like nfs), and the seed has to be the same across all processes

    The dataset is split to sub datasets of almost equal sizes and scattered
    to workers. To create a sub dataset, ``chainer.datasets.SubDataset`` is
    used.
    Args:
        dataset: A dataset (e.g., ``list``, ``numpy.ndarray``,
            ``chainer.datasets.TupleDataset``, ...).
        comm: ChainerMN communicator or MPI4py communicator.
        shuffle (bool): If ``True``, the order of examples is shuffled
            before being scattered.
        seed (int): Seed the generator used for the permutation of indexes.
            If an integer being convertible to 32 bit unsigned integers is
            specified, it is guaranteed that each sample
            in the given dataset always belongs to a specific subset.
            If ``None``, the permutation is changed randomly.
    Returns:
        Scattered dataset.
    """

    if hasattr(comm, 'mpi_comm'):
        comm = comm.mpi_comm
    assert hasattr(comm, 'send')
    assert hasattr(comm, 'recv')

    order = None
    n_total_samples = len(dataset)
    if shuffle is not None:
        order = numpy.random.RandomState(seed).permutation(
            n_total_samples)

    n_sub_samples = (n_total_samples + comm.size - 1) // comm.size

    b = n_total_samples * comm.rank // comm.size
    e = b + n_sub_samples

    return chainer.datasets.SubDataset(dataset, b, e, order)
