public class abstract ContaBancaria {
  private String senha;
  private int numero;
  private double saldo;
  public ContaBancaria(String senha, int numero, double saldo) {
    this.setSenha(senha);
    this.setNumero(numero);
    this.setSaldo(saldo);
  }
  public abstract void sacar(double valor);
  public abstract void depositar(double valor);
  public abstract void consultarExtrato();
  public void setSenha(String senha) {
    this.senha = senha;
  }
  public String getSenha() {
    return senha;
  }
  public void setNumero(int numero) {
    this.numero = numero;
  }
  public int getNumero() {
    return numero;
  }
  public void setSaldo(double saldo) {
    this.saldo = saldo;
  }
  public double getSaldo() {
    return saldo;
  }
  public static void alterarSenha(String NovaSenha) {
    Scanner sc=new Scanner(System.in);
    String atualSenha = sc.next();
    if (this.senha==atualSenha) {
      this.senha = NovaSenha;
    }
    System.out.println("A sua senha "+atualSenha+" foi modificada para "+senha+"");
  }
