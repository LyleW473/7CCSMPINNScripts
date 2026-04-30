import numpy as np
from sklearn import datasets

if __name__ == "__main__":

    iris = datasets.load_iris()
    classes = iris.target
    dataset = iris.data
    print(dataset)
    print(classes)

    vectors = np.array([
        [7.1, 3.8, 6.7, 2.5],
        [7.6, 2.0, 2.2, 0.4],
        [6.2, 3.1, 4.1, 2.4],
        [7.2, 2.6, 2.3, 0.4],
        [6.3, 2.7, 4.3, 0.5]
    ])
    
    for k in [1, 5]:
        answers = []
        for vector in vectors:
            
            # Find similarity scores (set to euclidean distance)
            similarities = []
            for vector_2, class_2 in zip(dataset, classes):
                differences = vector_2 - vector
                squared_distances = differences ** 2
                sum_squared_distances = np.sum(squared_distances)
                euclid_distance = np.sqrt(sum_squared_distances)
                similarities.append((euclid_distance, class_2))
        
            # Sort by the closest distance
            similarities.sort(key=lambda x: x[0])
    
            # Select the K nearest
            top_k = similarities[:k]

            # Select the majority class among the K nearest
            classes_k = [class_2 for _, class_2 in top_k]
            class_for_vector = max(set(classes_k), key=classes_k.count)
            answers.append(class_for_vector)
        print(f"K={k} | answers={[int(a) for a in answers]}")