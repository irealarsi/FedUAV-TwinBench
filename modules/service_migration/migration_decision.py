def should_migrate(predicted_queue, predicted_energy, migration_thresholds=None, verbose=False):
    """
    Decide whether to trigger service migration based on DT predictions.

    Args:
        predicted_queue (float): DT-predicted next queue length.
        predicted_energy (float): DT-predicted energy cost of current node.
        migration_thresholds (dict): Optional thresholds for triggering migration.
        verbose (bool): If True, prints decision info.

    Returns:
        bool: True if migration should be triggered.
    """
    if migration_thresholds is None:
        migration_thresholds = {
            "queue": 0.25,     # Lower threshold to trigger more frequently
            "energy": 0.20
        }

    queue_flag = predicted_queue >= migration_thresholds["queue"]
    energy_flag = predicted_energy >= migration_thresholds["energy"]
    should_migrate = queue_flag or energy_flag

    if verbose:
        print(f"[MIGRATION CHECK] Queue={predicted_queue:.3f} (>{migration_thresholds['queue']}?) → {queue_flag}, "
              f"Energy={predicted_energy:.3f} (>{migration_thresholds['energy']}?) → {energy_flag} → Migrate={should_migrate}")

    return should_migrate


if __name__ == "__main__":
    migrate = should_migrate(predicted_queue=0.22, predicted_energy=0.23, verbose=True)
    print("Trigger Migration:", migrate)
