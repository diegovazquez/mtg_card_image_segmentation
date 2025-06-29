import { nodeResolve } from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import babel from '@rollup/plugin-babel';
import terser from '@rollup/plugin-terser';
import copy from 'rollup-plugin-copy';
import gzipPlugin from 'rollup-plugin-gzip';

export default {
  input: 'src/app.js',
  output: [
    {
      file: 'dist/bundle.js',
      format: 'iife',
      name: 'CardSegmentationApp',
      sourcemap: true
    },
    {
      file: 'dist/bundle.min.js',
      format: 'iife',
      name: 'CardSegmentationApp',
      plugins: [terser()],
      sourcemap: true
    }
  ],
  plugins: [
    nodeResolve({
      browser: true,
      preferBuiltins: false
    }),
    commonjs(),
    babel({
      babelHelpers: 'bundled',
      exclude: 'node_modules/**',
      presets: [
        ['@babel/preset-env', {
          targets: {
            browsers: [
              'Safari >= 14',
              'Chrome >= 85',
              'Firefox >= 80',
              'Edge >= 85'
            ]
          },
          modules: false
        }]
      ]
    }),
    copy({
      targets: [
        { src: 'index.html', dest: 'dist' },
        { src: 'src/style.css', dest: 'dist' },
        { src: 'src/overlay.svg', dest: 'dist' },
        { src: 'models/*', dest: 'dist/models' }
      ]
    }),
    gzipPlugin({
      filter: /\.(js|css|html)$/
    })
  ]
};