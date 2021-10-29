import matplotlib.pyplot as plt


def write_histogram(file, histogram):
    """Create and write pitch histogram"""
    plt.figure()
    plt.hist(histogram.numpy(), bins=50)
    plt.title('Log pitch error distribution in voiced regions')
    plt.xlabel('Log pitch deviation')
    plt.ylabel('Frequency')
    plt.savefig(file)
    plt.close()
