import pymol
from pymol import cmd
pymol.finish_launching()
cmd.load("./7DK2_AB_C.pdb", "molecule")
cmd.h_add("molecule") # Adding hydrogen

cmd.wizard("mutagenesis")

# 获取所有链
chains = cmd.get_chains("molecule")
for chain in chains:
    # 获取每条链中的所有残基
    model = cmd.get_model("molecule and chain %s" % chain)
    residues = set(atom.resi for atom in model.atom)
    for resi in residues:
        # 选择并替换每个残基
        selection = "/%s//%s/%s" % ("molecule", chain, resi)
        try:
            cmd.select("temp_selection", selection)
            cmd.get_wizard().do_select("temp_selection")
            cmd.get_wizard().apply()
        except pymol.CmdException as e:
            print(f"Error selecting {selection}: {e}")

cmd.set_wizard()

# 保存修改后的结构到新文件
cmd.save("P_7DK2_AB_C_modified.pdb", "molecule")
