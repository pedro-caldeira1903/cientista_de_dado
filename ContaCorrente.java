public class ContaCorrente extends ContaBancaria {
  private int transacoes;
  public ContaCorrente(String senha, int numero, double saldo, int transacoes) {
    super(senha, numero, saldo);
  }
  public static void sacar(double valor){
    double novoSaldo=saldo - (valor*0.25);
    System.out.println("O valor que você vai sacar "+valor+" o saldo da conta vai ficar R$ "+novoSaldo+"");
  }
  public static void depositar(double valor){
    double novoSaldo=saldo + valor;
    System.out.println("O valor que você vai depositar "+valor+" o saldo da conta vai ficar R$ "+novoSaldo+"");
  }
  public static void consultarExtrato() {
    System.out.println("O valor de transições são "+transacoes+"");
  }
  public void setTransacoes(int transacoes) {
    this.transacoes= transacoes;
  }
  public int getTransacoes() {
    return transacoes;
  }
}
