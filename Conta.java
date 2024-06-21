public class Conta {
	public int numero;
	public String titular;
	public double saldo;
	public Conta(int numero, String titular, double saldo) {
		this.numero = numero;
		this.saldo = saldo;
		this.titular = titular;
	}
	public double depositar(double valor) {
		return valor + saldo;
	}
	public double sacar(double valor) {
		return saldo - (valor * 0.25);
	}
	public static void transferir(Conta c, double valor) {
		System.out.println("A quantia de R$ "+valor+" vai ser transferido para a conta do "+c.titular+"");
	}
	public boolean equals(Conta c) {
		if (this.numero == c.numero) {
			return true;
		}
		return false;
	}
}
