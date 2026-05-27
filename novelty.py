def novelty(similarity_values: list[float], threshold: float) -> float:
    if not similarity_values:
        return 0.0

    count = sum(1 for value in similarity_values if value < threshold)
    return count / len(similarity_values)
