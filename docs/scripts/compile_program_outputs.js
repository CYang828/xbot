const getProgramOutputs = require('../plugins/program_output.js');

console.info('Computing program outputs');
getProgramOutputs({
  docsDir: './docs',
  include: ['**.mdx', '**.md'],
  commandPrefix: 'XBOT_TELEMETRY_ENABLED=false poetry run',
});
