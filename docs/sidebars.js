module.exports = {
  docs: [
    {
      type: 'category',
      label: 'Getting Started',
      items: [
          'greeting',
      ],
    },
    {
      type: 'category',
      label: 'Guide',
      items: [
          'guide',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
          require('./docs/reference/sidebar.json'),
      ],
    },
  ],
};
