#include <stdio.h>
int fibonacci(int n) {
    if (n<=1) return n;
    else return fibonacci(n-2) + fibonacci(n-1);
}
int main(void) {
    int n;
    printf("Digite um número:");
    scanf("%d", &n);
    printf("%d\n", fibonacci(n));
    return 0;
}
