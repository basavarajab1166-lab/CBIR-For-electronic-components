#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void merge(int arr[], int l, int m, int r)
{
    int i = l, j = m + 1, k = 0;
    int n = r - l + 1;
    int *temp = (int *)malloc(n * sizeof(int));
    if (!temp)
    {
        perror("malloc");
        exit(1);
    }

    while (i <= m && j <= r)
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }
    while (i <= m)
        temp[k++] = arr[i++];
    while (j <= r)
        temp[k++] = arr[j++];

    for (i = l, k = 0; i <= r; i++, k++)
        arr[i] = temp[k];

    free(temp);
}

void sequentialMergeSort(int arr[], int l, int r)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;
        sequentialMergeSort(arr, l, m);
        sequentialMergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

/* parallelMergeSort with depth limit to avoid oversubscription */
void parallelMergeSort(int arr[], int l, int r, int depth)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;

        if (depth <= 0)
        {
            sequentialMergeSort(arr, l, m);
            sequentialMergeSort(arr, m + 1, r);
        }
        else
        {
#pragma omp parallel sections default(shared) num_threads(2)
            {
#pragma omp section
                parallelMergeSort(arr, l, m, depth - 1);

#pragma omp section
                parallelMergeSort(arr, m + 1, r, depth - 1);
            }
        }

        merge(arr, l, m, r);
    }
}

int main()
{
    int n = 100000; /* change smaller if you need */
    int *arr1 = malloc(n * sizeof(int));
    int *arr2 = malloc(n * sizeof(int));
    if (!arr1 || !arr2)
    {
        perror("malloc");
        return 1;
    }

    srand(42);
    for (int i = 0; i < n; i++)
    {
        arr1[i] = rand() % 1000000;
        arr2[i] = arr1[i];
    }

    omp_set_num_threads(omp_get_max_threads());

    double start = omp_get_wtime();
    sequentialMergeSort(arr1, 0, n - 1);
    double end = omp_get_wtime();
    printf("Sequential Merge Sort Time: %f seconds\n", end - start);

    int parallelDepth = 4; /* tune this: ~log2(num_cores) */
    start = omp_get_wtime();
    parallelMergeSort(arr2, 0, n - 1, parallelDepth);
    end = omp_get_wtime();
    printf("Parallel Merge Sort Time: %f seconds\n", end - start);

    /* quick validation */
    for (int i = 1; i < n; i++)
    {
        if (arr1[i - 1] > arr1[i])
        {
            printf("sequential not sorted at %d\n", i);
            break;
        }
        if (arr2[i - 1] > arr2[i])
        {
            printf("parallel not sorted at %d\n", i);
            break;
        }
    }

    free(arr1);
    free(arr2);
    return 0;
}