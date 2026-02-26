# Using VSCode
1. Download and install VSCode (https://code.visualstudio.com/).
2. Enable setting “remote.SSH.showLoginTerminal”
3. Install the Remote - SSH plugin (https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh).
4. Navigate to the left, click the Remote Explore icon, then click the plus (+) icon on the right to add new remote machines.
5. The command, for example, should be for caddy machine (but remember, you can't run any simulations on caddy machine; you will need FarmShare for that)
```
# VSCode SSH to caddy machine [for coding only]
ssh <YOUR SUID>@caddy.best.stanford.edu 
 
# or

# SSH to Farmshare [for coding and running simulations]
ssh <YOUR SUID>@login.farmshare.stanford.edu 
```
6. Enter the password in the terminal and follow the instructions. During this, you may be asked to provide 2-factor authentication.
7. Once it is set up, you can edit files using VSCode.

## Advanced IDE-like Features Through Clangd

You can install the [Clangd extension in VSCode](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) to get IDE-like features such as code completion, error checking, automatic formatting, tooltips on symbols and data types, and more.
This may be especially useful when there are multiple layers of typedefs and templates for different symbols in the codebase.

After installing the Clangd extension, create the following files in the root directory of your project:

```bash
touch .clangd # for configuring clangd search paths
touch compile_flags.txt # Leave empty for now
touch .clang-format # Optional, for code formatting
```

Go to https://code.stanford.edu/-/snippets/137 and copy the contents of the `.clangd` and `.clang-format` files into the respective files you just created.
Reload VSCode after making these changes. You should start seeing features like code completion and error checking.

<img src="images/clang_tooltip.png" alt="VSCode Clangd Tooltip Example"/>
