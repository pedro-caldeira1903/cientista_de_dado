public class ContaCorrente extends ContaBancaria {
  private int transacoes;
  public ContaCorrente(String senha, int numero, double saldo, int transacoes) {
    super(senha, numero, saldo);
  }
  public void setTransacoes(int transacoes) {
    this.transacoes = transacoes;
  }
  public int getTransacoes() {
    return transacoes;
  }
  public static void sacar(double valor) {
    double novoSaldo=saldo - (valor * 0.25);
    System.out.println("O valor que você sacou foi R$ "+valor+"e o saldo ficou R$ "+novoSaldo+".");
  }
  public static void depositar(double valor) {
    double novoSaldo=saldo + (valor * 0.25);
    System.out.println("O valor que você depositou foi R$ "+valor+"e o saldo ficou R$ "+novoSaldo+".");
  }
  public static void consultarExtrato() {
    System.out.println("O número de transações são "+transacoes+".");
  }
}
