public class abstract ContaBancaria {
  private String senha;
  private int numero;
  private double saldo;
  public ContaBancaria(String senha, int numero, double saldo) {
    this.senha = senha;
    this.numero = numero;
    this.saldo = saldo;
  }
  public abstract void sacar(double valor);
  public abstract void depositar(double valor);
  public abstract void consultarExtrato();
  public static void alteraSenha(String Novasenha) {
    if (Novasenha != this.senha) {
      this.senha = Novasenha;
    }
    System.out.println("A nova senha é "+ this.senha+"");
  }
}
