#include <stdio.h>
#include <cudnn.h>

int main() {
    size_t version = cudnnGetVersion();
    printf("cuDNN version: %zu\n", version);
    return 0;
}