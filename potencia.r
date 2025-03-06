potencia <- function(a, b) {
    if (b==1) {
        return(a)
    }
    if (b %%2 == 0) {
        return(potencia(a, b/2)^2)
    }
    else {
        return(a * potencia(a, b-1))
    }
}
