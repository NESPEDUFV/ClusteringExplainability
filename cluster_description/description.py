class Description: 

	@staticmethod
	def discrete_vars(col, percentage, value):
		return f"{col}: {percentage}% valores {value}"

	@staticmethod
	def continuos_vars(col, lower, upper):
		return f"{col}: 80% values between [{lower}, {upper}]"