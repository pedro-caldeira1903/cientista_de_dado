public class ContaPoupanca extends ContaBancaria {
  private double taxaRendimento;
  public ContaPoupanca(String senha, int numero, double saldo, double taxaRendimento) {
    super(senha, numero, saldo);
  }
  public void setTaxaRendimento(double taxaRendimento) {
    this.taxaRendimento = taxaRendimento;
  }
  public double getTaxaRendimento() {
    return taxaRendimento;
  }
  public static void sacar(double valor) {
    double novoSaldo=saldo - (valor * 0.25);
    System.out.println("O valor que você sacou foi de R$ "+valor+", o saldo é de R$ "+novoSaldo+".");
  }
  public static void depositar(double valor) {
    double novoSaldo=saldo + valor;
    System.out.println("O valor que você depositou foi de R$ "+valor+", o saldo é de R$ "+novoSaldo+".");
  }
  public static void consultarExtrato() {
    System.out.println("A taxa de redimento é de "+taxaRendimento+"%");
  }
}
